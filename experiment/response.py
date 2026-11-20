"""Purchase sampling and cached expected-revenue evaluation."""
import torch
from torch.distributions import Categorical


def sample_hotel_choice(choice_model, batch, selected, generator):
    b, _, d = batch["candidates"].shape
    maximum = max(1, int(selected.sum(1).max()))
    candidates = torch.zeros((b, maximum, d), device=selected.device)
    prop_ids = torch.full((b, maximum), -1, dtype=torch.long, device=selected.device)
    mask = torch.zeros((b, maximum), dtype=torch.bool, device=selected.device)
    positions = torch.full((b, maximum), -1, dtype=torch.long, device=selected.device)
    rows, columns = torch.nonzero(selected, as_tuple=True)
    local_positions = selected.cumsum(1).sub(1)[rows, columns]
    candidates[rows, local_positions] = batch["candidates"][rows, columns]
    prop_ids[rows, local_positions] = batch["prop_ids"][rows, columns]
    mask[rows, local_positions] = True
    positions[rows, local_positions] = columns
    with torch.no_grad():
        probabilities = choice_model.probabilities(
            batch["request"], candidates, prop_ids, batch["destination_ids"], mask
        )
        local = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    outside = local.eq(maximum) | ~mask.gather(1, local.clamp_max(maximum - 1)[:, None]).squeeze(1)
    original = positions.gather(1, local.clamp_max(maximum - 1)[:, None]).squeeze(1)
    original = torch.where(outside, torch.full_like(original, -1), original)
    revenue = batch["prices"].gather(1, original.clamp_min(0)[:, None]).squeeze(1)
    revenue = torch.where(original.ge(0), revenue, torch.zeros_like(revenue))
    return original, revenue


class CachedSingleChoiceEnvironment:
    """Single-choice environment using precomputed calibrated utilities."""

    def __init__(self, cache, price_scale=1.0, price_cap_raw=None):
        self.cache = cache
        self.price_scale = float(price_scale)
        self.price_cap_raw = (
            None if price_cap_raw is None else float(price_cap_raw)
        )

    def logits(self, batch):
        indices = batch["dataset_index"].detach().cpu().numpy()
        return self.cache.gather(indices, batch["prices"].device)

    @torch.no_grad()
    def outcomes(self, batch, selection_mask, sample):
        purchase_logits, outside_logit = self.logits(batch)
        valid_selection = selection_mask & batch["candidate_mask"].bool()
        candidate_logits = purchase_logits.masked_fill(~valid_selection, -1e9)
        choice_logits = torch.cat(
            (candidate_logits, outside_logit.unsqueeze(1)), dim=1
        )
        probability = torch.softmax(choice_logits, dim=1)
        item_probability = probability[:, :-1]
        prices = batch["prices"]
        if self.price_cap_raw is not None:
            prices = prices.clamp_max(self.price_cap_raw)
        if sample:
            choice = Categorical(probs=probability).sample()
            purchased = choice < prices.shape[1]
            raw_reward = torch.zeros(len(prices), device=prices.device)
            rows = torch.arange(len(prices), device=prices.device)[purchased]
            raw_reward[rows] = prices[rows, choice[purchased]]
            purchase_event = purchased.float()
        else:
            raw_reward = (item_probability * prices).sum(dim=1)
            purchase_event = item_probability.sum(dim=1)
        return {
            "raw_reward": raw_reward,
            "scaled_reward": raw_reward / self.price_scale,
            "purchase_event": purchase_event,
            "item_probability": item_probability,
        }
