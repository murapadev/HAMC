"""Entropy and weighted sampling over ``{option: weight}`` dicts."""

import logging
import math
import random
from functools import lru_cache
from typing import Dict, Optional, Set, Tuple


@lru_cache(maxsize=4096)
def _entropy_of(weights: Tuple[float, ...]) -> float:
    total = sum(weights)
    if total <= 0:
        return float("inf")
    return -sum((w / total) * math.log2(w / total) for w in weights if w > 0)


def shannon_entropy(weighted_options: Dict[str, float]) -> float:
    """Shannon entropy (bits) of the normalised weights; ``inf`` if empty."""
    if not weighted_options:
        return float("inf")
    return _entropy_of(tuple(sorted(weighted_options.values())))


def weighted_random_choice(weighted_options: Dict[str, float],
                           temperature: float = 1.0,
                           avoid_values: Optional[Set[str]] = None,
                           rng: Optional[random.Random] = None) -> str:
    """Pick a key with probability proportional to its weight.

    ``temperature`` > 1 flattens the distribution, < 1 sharpens it. Values in
    ``avoid_values`` are skipped unless nothing else is left. Draws come from
    ``rng`` (a :class:`random.Random`), so a seeded generator reproduces the
    same choices. Returns ``"ERROR"`` (and logs why) if no choice is possible.
    """
    rng = rng or random
    options = weighted_options
    if avoid_values:
        options = {k: v for k, v in weighted_options.items() if k not in avoid_values} or weighted_options
    try:
        keys = list(options)
        weights = [float(options[k]) for k in keys]
        if temperature != 1.0:
            weights = [w ** (1.0 / temperature) for w in weights]
        total = sum(weights)
        if not keys or total <= 0 or math.isnan(total):
            return "ERROR"
        threshold = rng.random() * total
        acc = 0.0
        for key, weight in zip(keys, weights):
            acc += weight
            if threshold < acc:
                return key
        return keys[-1]
    except (TypeError, ValueError) as e:
        logging.getLogger(__name__).warning(
            "weighted_random_choice failed (options=%d): %s", len(weighted_options), e
        )
        return "ERROR"
