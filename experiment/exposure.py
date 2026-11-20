"""Per-episode exposure allocation and accounting."""
import numpy as np
import torch


def allocate_capped_weighted_budget(occurrence, weight, target):
    """Integer weighted allocation with exact total and occurrence caps.

    Zero allocation is intentional: when an episode contains more distinct
    long-tail resources than the exposure budget, granting every resource one
    unit would silently inflate the experimental budget.
    """
    occurrence = np.asarray(occurrence, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.float64)
    target = min(max(int(target), 0), int(occurrence.sum()))
    budget = np.zeros_like(occurrence)
    remaining = target
    while remaining > 0:
        capacity = occurrence - budget
        active = capacity > 0
        if not active.any():
            break
        active_weight = np.where(active, np.maximum(weight, 1e-12), 0.0)
        quota = remaining * active_weight / active_weight.sum()
        addition = np.minimum(np.floor(quota).astype(np.int64), capacity)
        granted = int(addition.sum())
        if granted == 0:
            # Largest-remainder step; stable sort makes the allocation fully
            # deterministic when several resources have equal weight.
            candidates = np.flatnonzero(active)
            order = candidates[np.argsort(-quota[candidates], kind="stable")]
            take = min(remaining, len(order))
            addition[order[:take]] = 1
            granted = take
        budget += addition
        remaining -= granted
    if int(budget.sum()) != target:
        raise RuntimeError("weighted budget allocation did not conserve total")
    return budget


class SequentialBudgetBatch:
    def __init__(
        self,
        session_dataset,
        item_store,
        frequency_store,
        episode_session_indices,
        episode_time_mask,
        cluster_ids,
        resource_level,
        rho,
        max_assortment,
        device,
    ):
        self.dataset = session_dataset
        self.item_store = item_store
        self.frequency_store = frequency_store
        self.resource_level = resource_level
        self.device = device
        self.resource_ids = []
        self.initial = []
        self.remaining = []
        self.total_initial = 0.0
        self.total_consumed = 0.0
        for row, time_mask, cluster_id in zip(
            episode_session_indices, episode_time_mask, cluster_ids
        ):
            active_indices = row[time_mask]
            item_ids = session_dataset.arrays["item_ids"][active_indices]
            mask = session_dataset.arrays["candidate_mask"][active_indices].astype(bool)
            safe = np.where(
                (item_ids >= 0) & (item_ids < item_store.length), item_ids, 0
            )
            mask &= item_store.valid[safe].astype(bool)
            if resource_level == "seller":
                resources = np.asarray(item_store.seller[safe], dtype=np.int64)
            else:
                resources = item_ids.astype(np.int64)
            values, occurrence = np.unique(resources[mask & (resources > 0)], return_counts=True)
            train_count = frequency_store.counts_for(cluster_id, values)
            weight = np.sqrt(np.maximum(train_count, 1.0))
            target = max(1, int(round(float(rho) * int(time_mask.sum()) * max_assortment)))
            budget = allocate_capped_weighted_budget(
                occurrence, weight, target
            )
            ids_tensor = torch.from_numpy(values).long().to(device)
            initial_tensor = torch.from_numpy(budget).float().to(device)
            self.resource_ids.append(ids_tensor)
            self.initial.append(initial_tensor)
            self.remaining.append(initial_tensor.clone())
            self.total_initial += float(initial_tensor.sum().item())

    def _batch_resource_ids(self, batch):
        if self.resource_level == "seller":
            return batch["seller"]
        return batch["item_ids"]

    def candidate_state(self, batch):
        resources = self._batch_resource_ids(batch)
        ratio = torch.zeros_like(batch["prices"], dtype=torch.float32)
        available = torch.zeros_like(batch["candidate_mask"], dtype=torch.bool)
        for row in range(len(resources)):
            ids = self.resource_ids[row]
            if len(ids) == 0:
                continue
            current = resources[row]
            positions = torch.searchsorted(ids, current)
            safe = positions.clamp(max=len(ids) - 1)
            matched = (current > 0) & (ids[safe] == current)
            remaining = self.remaining[row][safe]
            initial = self.initial[row][safe].clamp_min(1.0)
            ratio[row] = torch.where(matched, remaining / initial, 0.0)
            available[row] = matched & (remaining > 0)
        return ratio, available

    def consume(self, batch, selection_mask, active_rows=None):
        resources = self._batch_resource_ids(batch)
        if active_rows is None:
            active_rows = torch.ones(len(resources), dtype=torch.bool, device=resources.device)
        for row in range(len(resources)):
            if not bool(active_rows[row]):
                continue
            ids = self.resource_ids[row]
            selected_ids = resources[row][selection_mask[row] & (resources[row] > 0)]
            if len(ids) == 0 or len(selected_ids) == 0:
                continue
            positions = torch.searchsorted(ids, selected_ids).clamp(max=len(ids) - 1)
            matched = ids[positions] == selected_ids
            positions = positions[matched]
            decrement = torch.zeros_like(self.remaining[row])
            decrement.scatter_add_(0, positions, torch.ones_like(positions, dtype=torch.float32))
            actual = torch.minimum(decrement, self.remaining[row])
            self.remaining[row] -= actual
            self.total_consumed += float(actual.sum().item())

    def utilization(self):
        return self.total_consumed / max(self.total_initial, 1.0)
