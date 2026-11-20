"""Business outcome denominators for reported policy evaluations."""
from collections.abc import Iterable
from statistics import mean, stdev


def business_outcome(rewards: Iterable[float], *, unit: str, count: int) -> float:
    """Sum daily totals or request rewards and apply the reporting denominator.

    For hotel outcomes, supply booking revenues. For product search, supply
    expected GMV in the reported price units. Count all test days or requests.
    """
    values = list(rewards)
    if count <= 0 or len(values) != count:
        raise ValueError('Supply one outcome for every evaluated day or request')
    scale = {'revenue/day': 1, 'revenue/search': 1, 'gmv/1000': 1000}[unit]
    return scale * sum(values) / count


def across_replicates(values: Iterable[float]) -> tuple[float, float]:
    """Summarize independent replicate outcomes with the sample SD."""
    scores = list(values)
    if len(scores) < 2:
        raise ValueError('Sample SD requires at least two replicates')
    return mean(scores), stdev(scores)
