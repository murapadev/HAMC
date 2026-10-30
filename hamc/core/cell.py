import random

from .entropy import shannon_entropy, weighted_random_choice


class Cell:
    """One grid position: the values it may still take and their weights.

    ``possible`` maps each remaining value to its weight. The solver never
    mutates that dict in place (it assigns a new one), which lets it keep
    references to old domains on its undo trail.
    """

    __slots__ = ("possible", "collapsed_value")

    def __init__(self, possible_values: dict[str, float]):
        if not possible_values:
            raise ValueError("Cannot initialize cell with empty possibilities")
        if any(w < 0 for w in possible_values.values()):
            raise ValueError("All weights must be non-negative")
        self.possible: dict[str, float] = {k: v for k, v in possible_values.items() if v > 0}
        if not self.possible:
            raise ValueError("After filtering zero weights, no valid options remain")
        self.collapsed_value: str | None = None

    def entropy(self) -> float:
        """Shannon entropy of the remaining options; -1 once collapsed."""
        if self.collapsed_value is not None:
            return -1
        return shannon_entropy(self.possible)

    def collapse(self, avoid_values: set[str] | None = None, temperature: float = 1.0,
                 context: dict[str, dict[str, str]] | None = None,
                 rng: random.Random | None = None) -> str:
        """Pick one value by weight (biased towards collapsed neighbours).

        Returns the chosen value, or ``"ERROR"`` if nothing can be chosen.
        """
        options = self.possible
        if context and context.get("neighbor_values"):
            options = self._adjust_for_neighbors(options, context["neighbor_values"])
        value = weighted_random_choice(options, temperature, avoid_values, rng=rng)
        if value == "ERROR":
            return value
        self.collapsed_value = value
        self.possible = {value: 1.0}
        return value

    @staticmethod
    def _adjust_for_neighbors(options: dict[str, float], neighbors: dict[str, str]) -> dict[str, float]:
        """Boost values already used by neighbours so tiles form clusters
        instead of speckled noise (+0.7 per matching neighbour, +0.3 more when
        two or more agree)."""
        counts: dict[str, int] = {}
        for value in neighbors.values():
            if value:
                counts[value] = counts.get(value, 0) + 1
        adjusted = dict(options)
        for option, n in counts.items():
            if option in adjusted:
                adjusted[option] *= 1.0 + n * 0.7 + (0.3 if n >= 2 else 0.0)
        return adjusted
