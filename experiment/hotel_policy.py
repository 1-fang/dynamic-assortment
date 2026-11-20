"""Dynamic Set-Transformer encoder and autoregressive Pointer-Transformer A2C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.distributions import Categorical

class StateProgram(Protocol):
    def features(self, request, candidates, prices, inventory, time, mask): ...


@dataclass(frozen=True)
class PolicyConfig:
    request_dim: int
    candidate_dim: int
    hidden_dim: int = 128
    num_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1
    slate_size: int = 4
    destination_hash_size: int = 8192
    property_hash_size: int = 65536
    global_lesr_dim: int = 8
    candidate_lesr_dim: int = 4

    def validate(self):
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (self.global_lesr_dim, self.candidate_lesr_dim) != (8, 4):
            raise ValueError("LESR dimensions must remain 8/4 during search")


@dataclass
class PolicyOutput:
    selected: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class AssortmentTransformer(nn.Module):
    def __init__(self, config: PolicyConfig, lesr: StateProgram) -> None:
        super().__init__()
        config.validate()
        self.config, self.lesr = config, lesr
        h = config.hidden_dim
        self.request_mlp = nn.Sequential(
            nn.Linear(config.request_dim, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, h)
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(config.candidate_dim + 1 + config.candidate_lesr_dim, h),
            nn.GELU(), nn.LayerNorm(h), nn.Linear(h, h),
        )
        self.destination_embedding = nn.Embedding(config.destination_hash_size, h)
        self.property_embedding = nn.Embedding(config.property_hash_size, h)
        self.global_lesr = nn.Sequential(
            nn.Linear(config.global_lesr_dim, h), nn.GELU(), nn.LayerNorm(h)
        )
        self.feature_gate = nn.Parameter(torch.tensor(0.0))

        encoder_layer = nn.TransformerEncoderLayer(
            h, config.num_heads, config.feedforward_dim, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.encoder_layers, norm=nn.LayerNorm(h)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            h, config.num_heads, config.feedforward_dim, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, config.decoder_layers, norm=nn.LayerNorm(h)
        )
        self.start_token = nn.Parameter(torch.empty(1, 1, h))
        self.stop_token = nn.Parameter(torch.empty(1, 1, h))
        self.decoder_positions = nn.Parameter(torch.empty(1, config.slate_size + 1, h))
        self.query = nn.Linear(h, h, bias=False)
        self.key = nn.Linear(h, h, bias=False)
        self.stop_head = nn.Linear(h, 1)
        self.critic = nn.Sequential(
            nn.Linear(2 * h, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, 1)
        )
        for parameter in (self.start_token, self.stop_token, self.decoder_positions):
            nn.init.normal_(parameter, std=0.02)

    @staticmethod
    def _mean(values, mask):
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(1) / weights.sum(1).clamp_min(1)

    def encode(self, request, candidates, prop_ids, destination_ids, prices, inventory, time, mask):
        global_f, candidate_f = self.lesr.features(
            request, candidates, prices, inventory, time, mask
        )
        gate = torch.tanh(self.feature_gate)
        request_token = self.request_mlp(request)
        request_token = request_token + self.destination_embedding(
            destination_ids.abs().remainder(self.config.destination_hash_size)
        )
        request_token = request_token + gate * self.global_lesr(global_f)
        candidate_input = torch.cat(
            (candidates, inventory.unsqueeze(-1), gate * candidate_f), dim=-1
        )
        candidate_tokens = self.candidate_mlp(candidate_input)
        candidate_tokens = candidate_tokens + self.property_embedding(
            prop_ids.abs().remainder(self.config.property_hash_size)
        )
        tokens = torch.cat((request_token.unsqueeze(1), candidate_tokens), dim=1)
        valid = torch.cat((
            torch.ones((request.shape[0], 1), dtype=torch.bool, device=request.device), mask
        ), dim=1)
        memory = self.encoder(tokens, src_key_padding_mask=~valid)
        contextual_candidates = memory[:, 1:]
        pooled = self._mean(contextual_candidates, mask)
        value = self.critic(torch.cat((memory[:, 0], pooled), dim=-1)).squeeze(-1)
        return memory, valid, contextual_candidates, value

    def _decoder_logits(self, history, memory, memory_valid, candidate_tokens, available):
        length = history.shape[1]
        target = history + self.decoder_positions[:, :length]
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=history.device), diagonal=1
        )
        decoded = self.decoder(
            target, memory, tgt_mask=causal, memory_key_padding_mask=~memory_valid
        )[:, -1]
        query = self.query(decoded).unsqueeze(-1)
        scores = torch.bmm(self.key(candidate_tokens), query).squeeze(-1) / (
            self.config.hidden_dim ** 0.5
        )
        scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
        return torch.cat((scores, self.stop_head(decoded)), dim=1)

    def first_action_logits(
        self, request, candidates, prop_ids, destination_ids, prices, inventory, time, mask
    ):
        memory, memory_valid, candidate_tokens, _ = self.encode(
            request, candidates, prop_ids, destination_ids, prices, inventory, time, mask
        )
        history = self.start_token.expand(request.shape[0], -1, -1)
        return self._decoder_logits(
            history, memory, memory_valid, candidate_tokens, mask & inventory.gt(0)
        )

    def forward(
        self, request, candidates, prop_ids, destination_ids, prices, inventory, time, mask,
        deterministic: bool = False,
    ) -> PolicyOutput:
        memory, memory_valid, candidate_tokens, value = self.encode(
            request, candidates, prop_ids, destination_ids, prices, inventory, time, mask
        )
        batch, count = request.shape[0], candidates.shape[1]
        selected = torch.zeros((batch, count), dtype=torch.bool, device=request.device)
        alive = torch.ones(batch, dtype=torch.bool, device=request.device)
        log_prob = torch.zeros(batch, dtype=request.dtype, device=request.device)
        entropy = torch.zeros_like(log_prob)
        history = self.start_token.expand(batch, -1, -1)
        base_available = mask & inventory.gt(0)
        for _ in range(self.config.slate_size):
            if not bool(alive.any()):
                break
            logits = self._decoder_logits(
                history, memory, memory_valid, candidate_tokens, base_available & ~selected
            )
            logits = torch.where(alive[:, None], logits, torch.zeros_like(logits))
            distribution = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else distribution.sample()
            log_prob = log_prob + torch.where(
                alive, distribution.log_prob(action), torch.zeros_like(log_prob)
            )
            entropy = entropy + torch.where(
                alive, distribution.entropy(), torch.zeros_like(entropy)
            )
            stop = action.eq(count)
            choose = alive & ~stop
            safe = action.clamp_max(count - 1)
            chosen_tokens = candidate_tokens[torch.arange(batch, device=request.device), safe]
            append = torch.where(
                choose[:, None], chosen_tokens, self.stop_token.expand(batch, -1, -1)[:, 0]
            )
            history = torch.cat((history, append.unsqueeze(1)), dim=1)
            newly_selected = torch.nn.functional.one_hot(safe, num_classes=count).to(torch.bool)
            selected = selected | (newly_selected & choose[:, None])
            alive = alive & ~stop
        return PolicyOutput(selected, log_prob, entropy, value)
