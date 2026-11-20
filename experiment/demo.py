"""Run a CPU smoke example with constructed tensors."""
import torch

from .exposure import allocate_capped_weighted_budget
from .hotel_policy import AssortmentTransformer, PolicyConfig
from .metrics import business_outcome
from .product_policy import PointerActorCritic
from .response import CachedSingleChoiceEnvironment
from .training import A2CLossConfig, a2c_loss


class ZeroState:
    def features(self, request, candidates, prices, inventory, time, mask):
        return request.new_zeros((len(request), 8)), candidates.new_zeros((*mask.shape, 4))


class ExampleResponseCache:
    def gather(self, indices, device):
        return torch.zeros((len(indices), 3), device=device), torch.zeros(len(indices), device=device)


def main():
    torch.manual_seed(17)
    batch_size, width = 2, 3
    available = torch.tensor([[True, True, False], [False, False, False]])
    prices = torch.tensor([[5.0, 7.0, 11.0], [3.0, 6.0, 9.0]])
    inventory = available.float()
    hotel = AssortmentTransformer(
        PolicyConfig(request_dim=3, candidate_dim=4, hidden_dim=16,
                     feedforward_dim=32, encoder_layers=1, decoder_layers=1,
                     destination_hash_size=8, property_hash_size=8, slate_size=2),
        ZeroState(),
    )
    result = hotel(torch.zeros(batch_size, 3), torch.zeros(batch_size, width, 4),
                   torch.ones_like(available, dtype=torch.long), torch.ones(batch_size, dtype=torch.long),
                   prices, inventory, torch.zeros(batch_size), available)
    assert not (result.selected & ~available).any()
    assert (result.selected.sum(1) <= 2).all()
    loss = a2c_loss([{'value': [result.value[0]], 'logp': [result.log_prob[0]],
                      'entropy': [result.entropy[0]], 'reward': [torch.tensor(1.)]}], A2CLossConfig())
    assert torch.isfinite(loss)
    loss.backward()

    product = PointerActorCritic({
        'model': {'embedding_dim': 4, 'd_model': 16, 'max_assortment': 2,
                  'num_heads': 2, 'ff_dim': 32, 'dropout': 0.0, 'num_layers': 1},
        'vocab': dict.fromkeys(('user', 'query', 'entrance', 'item', 'title', 'brand', 'seller', 'category'), 8),
    })
    item = torch.ones(batch_size, width, dtype=torch.long)
    batch = {'user_bucket': item[:, 0], 'query_tokens': item, 'entrance_bucket': item[:, 0],
             'user_features': torch.zeros(batch_size, 6), 'item_ids': item, 'title_tokens': item[:, :, None],
             'brand': item, 'seller': item, 'category': item[:, :, None], 'item_stats': torch.zeros(batch_size, width, 3),
             'candidate_mask': available, 'prices': prices, 'dataset_index': torch.arange(batch_size)}
    selection = product(batch, greedy=True).selection_mask
    assert not (selection & ~available).any()
    assert (selection.sum(1) <= 2).all()
    rewards = CachedSingleChoiceEnvironment(ExampleResponseCache()).outcomes(batch, available, sample=False)
    assert torch.allclose(rewards['scaled_reward'], torch.tensor([4., 0.]))
    assert business_outcome(rewards['scaled_reward'].tolist(), unit='gmv/1000', count=2) == 2000
    budget = allocate_capped_weighted_budget([1, 3, 2], [1, 1, 2], 4)
    assert sum(budget) == 4 and all(budget <= [1, 3, 2])
    print('Hotel and product policy, A2C, response and budget smoke checks passed.')


if __name__ == '__main__':
    main()
