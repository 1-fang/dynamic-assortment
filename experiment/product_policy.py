"""Transformer candidate encoder with a permutation-aware Pointer Actor-Critic."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .catalog_encoder import NativeCandidateEncoder, masked_mean


@dataclass
class PointerOutput:
    selection_mask: torch.Tensor
    selection_order: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class PointerActorCritic(nn.Module):
    def __init__(
        self, config, ignore_budget_state=False, ignore_progress=False
    ):
        super().__init__()
        model = config["model"]
        d_model = int(model["d_model"])
        self.max_assortment = int(model["max_assortment"])
        self.ignore_budget_state = ignore_budget_state
        self.ignore_progress = ignore_progress
        self.category_vocab = int(config["vocab"]["category"])
        self.encoder = NativeCandidateEncoder(config)
        self.budget_projection = nn.Linear(1, d_model)
        self.progress_projection = nn.Linear(2, d_model)
        self.cluster_embedding = nn.Embedding(
            self.category_vocab, d_model, padding_idx=0
        )
        self.candidate_key = nn.Linear(d_model, d_model, bias=False)
        self.state_query = nn.Linear(d_model, d_model, bias=False)
        self.pointer_score = nn.Linear(d_model, 1, bias=False)
        self.selected_summary = nn.Linear(d_model, d_model)
        self.stop_head = nn.Linear(d_model, 1)
        self.critic = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        nn.init.constant_(self.stop_head.bias, -1.0)

    def encode_state(self, batch):
        candidates, context = self.encoder(batch)
        if not self.ignore_budget_state and "budget_ratio" in batch:
            candidates = candidates + self.budget_projection(
                batch["budget_ratio"].unsqueeze(-1)
            )
        if not self.ignore_progress and "progress" in batch:
            context = context + self.progress_projection(batch["progress"])
        if "cluster_id" in batch:
            cluster = torch.where(
                batch["cluster_id"] > 0,
                torch.remainder(batch["cluster_id"], self.category_vocab - 1) + 1,
                torch.zeros_like(batch["cluster_id"]),
            )
            context = context + self.cluster_embedding(cluster)
        return candidates, context

    def value(self, batch):
        candidates, context = self.encode_state(batch)
        valid = batch["candidate_mask"].bool()
        pooled = masked_mean(candidates, valid, dim=1)
        return self.critic(torch.cat((pooled, context), dim=-1)).squeeze(-1)

    def forward(self, batch, greedy=False):
        candidates, context = self.encode_state(batch)
        valid = batch["candidate_mask"].bool()
        batch_size, candidate_count, _ = candidates.shape
        pooled = masked_mean(candidates, valid, dim=1)
        value = self.critic(torch.cat((pooled, context), dim=-1)).squeeze(-1)

        selected = torch.zeros_like(valid)
        order = torch.full(
            (batch_size, self.max_assortment),
            -1,
            dtype=torch.long,
            device=candidates.device,
        )
        active = torch.ones(
            batch_size, dtype=torch.bool, device=candidates.device
        )
        log_probability = torch.zeros(batch_size, device=candidates.device)
        entropy = torch.zeros(batch_size, device=candidates.device)
        candidate_keys = self.candidate_key(candidates)

        for step in range(self.max_assortment):
            selected_mean = masked_mean(candidates, selected, dim=1)
            pointer_state = context + self.selected_summary(selected_mean)
            scores = self.pointer_score(
                torch.tanh(
                    candidate_keys + self.state_query(pointer_state).unsqueeze(1)
                )
            ).squeeze(-1)
            live_mask = (~valid) | selected
            scores = scores.masked_fill(live_mask.clone(), -1e9)
            stop_score = self.stop_head(pointer_state)
            logits = torch.cat((scores, stop_score), dim=1)
            if (~active).any():
                logits = logits.clone()
                logits[~active, :-1] = -1e9
                logits[~active, -1] = 0.0
            distribution = Categorical(logits=logits)
            choice = logits.argmax(dim=1) if greedy else distribution.sample()
            was_active = active.clone()
            log_probability = log_probability + (
                distribution.log_prob(choice) * was_active.float()
            )
            entropy = entropy + distribution.entropy() * was_active.float()
            is_item = choice < candidate_count
            chosen_rows = torch.arange(batch_size, device=choice.device)[is_item]
            chosen_items = choice[is_item]
            selected[chosen_rows, chosen_items] = True
            order[chosen_rows, step] = chosen_items
            active = active & is_item
        return PointerOutput(
            selection_mask=selected,
            selection_order=order,
            log_probability=log_probability,
            entropy=entropy,
            value=value,
        )
