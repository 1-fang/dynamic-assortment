"""Content-aware user/query/item and candidate-set encoders."""

import torch
import torch.nn as nn

def masked_mean(values, mask, dim):
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator



class MeanTokenEmbedding(nn.Module):
    def __init__(self, vocab_size, dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        return masked_mean(embedded, tokens != 0, dim=-2)


class ContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab = config["vocab"]
        model = config["model"]
        embedding_dim = int(model["embedding_dim"])
        d_model = int(model["d_model"])
        self.user_embedding = nn.Embedding(
            int(vocab["user"]), embedding_dim, padding_idx=0
        )
        self.query_embedding = MeanTokenEmbedding(
            int(vocab["query"]), embedding_dim
        )
        self.entrance_embedding = nn.Embedding(
            int(vocab["entrance"]), embedding_dim, padding_idx=0
        )
        self.numeric = nn.Sequential(
            nn.Linear(6, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )
        self.projection = nn.Sequential(
            nn.Linear(4 * embedding_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, batch):
        user = self.user_embedding(batch["user_bucket"])
        query = self.query_embedding(batch["query_tokens"])
        entrance = self.entrance_embedding(batch["entrance_bucket"])
        numeric = self.numeric(
            torch.log1p(batch["user_features"].clamp_min(0.0))
        )
        return self.projection(torch.cat((user, query, entrance, numeric), dim=-1))


class ItemEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab = config["vocab"]
        model = config["model"]
        embedding_dim = int(model["embedding_dim"])
        d_model = int(model["d_model"])
        self.item_vocab = int(vocab["item"])
        self.brand_vocab = int(vocab["brand"])
        self.seller_vocab = int(vocab["seller"])
        self.category_vocab = int(vocab["category"])
        self.item_embedding = nn.Embedding(
            self.item_vocab, embedding_dim, padding_idx=0
        )
        self.title_embedding = MeanTokenEmbedding(
            int(vocab["title"]), embedding_dim
        )
        self.brand_embedding = nn.Embedding(
            int(vocab["brand"]), embedding_dim, padding_idx=0
        )
        self.seller_embedding = nn.Embedding(
            int(vocab["seller"]), embedding_dim, padding_idx=0
        )
        self.category_embedding = nn.Embedding(
            int(vocab["category"]), embedding_dim, padding_idx=0
        )
        self.numeric = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )
        self.projection = nn.Sequential(
            nn.Linear(6 * embedding_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, batch):
        item_ids = batch["item_ids"]
        item_bucket = torch.where(
            item_ids > 0,
            torch.remainder(item_ids, self.item_vocab - 1) + 1,
            torch.zeros_like(item_ids),
        )
        item = self.item_embedding(item_bucket)
        title = self.title_embedding(batch["title_tokens"])
        brand_ids = batch["brand"]
        seller_ids = batch["seller"]
        brand_bucket = torch.where(
            brand_ids > 0,
            torch.remainder(brand_ids, self.brand_vocab - 1) + 1,
            torch.zeros_like(brand_ids),
        )
        seller_bucket = torch.where(
            seller_ids > 0,
            torch.remainder(seller_ids, self.seller_vocab - 1) + 1,
            torch.zeros_like(seller_ids),
        )
        brand = self.brand_embedding(brand_bucket)
        seller = self.seller_embedding(seller_bucket)
        category_tokens = batch["category"]
        category_tokens = torch.where(
            category_tokens > 0,
            torch.remainder(category_tokens, self.category_vocab - 1) + 1,
            torch.zeros_like(category_tokens),
        )
        category = masked_mean(
            self.category_embedding(category_tokens),
            category_tokens != 0,
            dim=-2,
        )
        numeric_values = torch.cat(
            (
                torch.log1p(batch["prices"].clamp_min(0.0)).unsqueeze(-1),
                torch.log1p(batch["item_stats"].clamp_min(0.0)),
            ),
            dim=-1,
        )
        numeric = self.numeric(numeric_values)
        return self.projection(
            torch.cat((item, title, brand, seller, category, numeric), dim=-1)
        )


class NativeCandidateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = config["model"]
        d_model = int(model["d_model"])
        self.context_encoder = ContextEncoder(config)
        self.item_encoder = ItemEncoder(config)
        self.context_to_candidate = nn.Linear(d_model, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(model["num_heads"]),
            dim_feedforward=int(model["ff_dim"]),
            dropout=float(model["dropout"]),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=int(model["num_layers"])
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, batch):
        mask = batch["candidate_mask"].bool()
        # Transformer attention cannot consume a row where every key is masked.
        # Keep one zero/padded token internally; downstream actors still use the
        # original mask and can only choose STOP for that row.
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        context = self.context_encoder(batch)
        candidates = self.item_encoder(batch)
        candidates = candidates + self.context_to_candidate(context).unsqueeze(1)
        encoded = self.transformer(
            candidates, src_key_padding_mask=~safe_mask
        )
        encoded = self.output_norm(encoded)
        encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0.0)
        return encoded, context
