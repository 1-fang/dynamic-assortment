"""Actor–critic loss for completed rollouts."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class A2CLossConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01


def a2c_loss(traces, config):
    logps, values, entropies, advantages, returns = [], [], [], [], []
    for trace in traces:
        if not trace["value"]:
            continue
        v = torch.stack(trace["value"])
        r = torch.stack(trace["reward"])
        adv = torch.zeros_like(v)
        gae = torch.zeros((), device=v.device)
        next_value = torch.zeros((), device=v.device)
        for t in reversed(range(len(v))):
            delta = r[t] + config.gamma * next_value - v[t].detach()
            gae = delta + config.gamma * config.gae_lambda * gae
            adv[t] = gae
            next_value = v[t].detach()
        logps.append(torch.stack(trace["logp"]))
        values.append(v)
        entropies.append(torch.stack(trace["entropy"]))
        advantages.append(adv)
        returns.append(adv + v.detach())
    logp, value, entropy = map(torch.cat, (logps, values, entropies))
    advantage, target = map(torch.cat, (advantages, returns))
    normalized_advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    actor = -(logp * normalized_advantage.detach()).mean()
    critic = torch.nn.functional.smooth_l1_loss(value, target.detach())
    return actor + config.value_coef * critic - config.entropy_coef * entropy.mean()
