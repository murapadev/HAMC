"""Merge user weight overrides into the default tables."""

from collections.abc import Collection, Mapping


def merge_weights(defaults: Mapping[str, float], overrides: Mapping[str, float] | None,
                  known: Collection[str], kind: str) -> dict[str, float]:
    """``defaults`` updated with ``overrides``.

    Overrides may change existing weights, add values from ``known`` or set a
    weight to 0 to drop a value. Unknown names raise ``ValueError`` so typos
    don't pass silently.
    """
    merged = dict(defaults)
    for name, weight in (overrides or {}).items():
        if name not in known:
            raise ValueError(f"unknown {kind} {name!r}; expected one of {sorted(known)}")
        if weight < 0:
            raise ValueError(f"{kind} weight for {name!r} must be >= 0")
        merged[name] = float(weight)
    if not any(w > 0 for w in merged.values()):
        raise ValueError(f"at least one {kind} needs a positive weight")
    return merged
