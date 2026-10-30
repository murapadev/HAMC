"""End-to-end generation: global regions -> intermediate blocks -> local tiles."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .generators.global_generator import GlobalGenerator
from .generators.intermediate_generator import IntermediateGenerator
from .generators.local_generator import LocalGenerator

Grid = List[List[str]]

#: Smallest block side that can hold every block layout (see LocalGenerator).
MIN_LOCAL = 3
Observer = Callable[[Dict[str, Any]], None]


class GenerationError(RuntimeError):
    """A layer could not be generated (or validated, in strict mode)."""

    def __init__(self, message: str, issues: Optional[List[str]] = None):
        super().__init__(message)
        self.issues = issues or [message]


class GenerationCancelled(RuntimeError):
    """Raised when the ``cancel`` callback asks the run to stop."""


@dataclass
class HAMCMap:
    """Result of one run. All grids are row-major lists of strings."""

    width: int
    height: int
    subgrid: int
    local: int
    seed: Optional[int]
    regions: Grid
    blocks: Grid
    tilemap: Grid
    valid: bool = True
    issues: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    block_attempts: int = 0

    def block_neighbors(self, r: int, c: int) -> Dict[str, str]:
        return block_neighbors(self.blocks, r, c)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)


def block_neighbors(blocks: Grid, r: int, c: int) -> Dict[str, str]:
    """Types of the blocks around ``(r, c)``, keyed by side."""
    out: Dict[str, str] = {}
    for side, dr, dc in (("top", -1, 0), ("bottom", 1, 0), ("left", 0, -1), ("right", 0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(blocks) and 0 <= nc < len(blocks[0]) and blocks[nr][nc]:
            out[side] = blocks[nr][nc]
    return out


def _tagged(observer: Optional[Observer], **tags: Any) -> Optional[Observer]:
    if observer is None:
        return None

    def relay(event: Dict[str, Any]) -> None:
        event.update(tags)
        observer(event)
    return relay


def generate_map(
    width: int = 4,
    height: int = 3,
    subgrid: int = 2,
    local: int = 4,
    *,
    seed: Optional[int] = None,
    observer: Optional[Observer] = None,
    step_delay: float = 0.0,
    cancel: Optional[Callable[[], bool]] = None,
    strict: bool = True,
) -> HAMCMap:
    """Generate a full map.

    ``width`` x ``height`` is the size of the region grid; each region holds
    ``subgrid`` x ``subgrid`` blocks and each block ``local`` x ``local``
    tiles, so the tilemap is ``height*subgrid*local`` rows by
    ``width*subgrid*local`` columns.

    ``seed`` makes the run reproducible; when omitted a random seed is drawn
    and stored in ``HAMCMap.seed``.

    ``observer`` receives every generator event tagged with its ``level``
    (``global``, ``intermediate`` or ``local``) plus the pipeline events
    ``local_init``, ``local_block_start`` and ``local_block_complete``.

    With ``strict`` (the default) any layer or block that fails validation
    raises :class:`GenerationError`; otherwise the map is returned with
    ``valid=False`` and the problems listed in ``issues``.
    """
    if subgrid < 1:
        raise ValueError(f"subgrid must be >= 1 (got {subgrid})")
    if local < MIN_LOCAL:
        raise ValueError(
            f"local must be >= {MIN_LOCAL} (got {local}): a building block needs a roof row, "
            "a facade row and a ground floor with a door")
    issues: List[str] = []
    timings: Dict[str, float] = {}

    def check_cancel() -> None:
        if cancel is not None and cancel():
            raise GenerationCancelled("generation cancelled")

    # One private Random per run: the same seed gives the same map, and runs
    # in other threads don't touch this state. Without a seed one is drawn
    # and recorded in the result so the map can be reproduced later.
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)

    t0 = time.perf_counter()
    G = GlobalGenerator(width, height, rng=rng)
    G.set_observer(_tagged(observer, level="global"), step_delay=step_delay)
    G.initialize()
    if not G.collapse():
        raise GenerationError("global layer failed")
    regions = G.tile_grid()
    timings["global"] = time.perf_counter() - t0
    check_cancel()

    t0 = time.perf_counter()
    I = IntermediateGenerator(G, subgrid_size=subgrid, rng=rng)
    I.set_observer(_tagged(observer, level="intermediate"), step_delay=step_delay * 0.6)
    if not I.collapse():
        issues.append("intermediate layer failed validation")
        if strict or any(c.collapsed_value is None for row in I.cells for c in row):
            raise GenerationError(issues[0], issues)
    blocks = I.tile_grid()
    timings["intermediate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows, cols = len(blocks), len(blocks[0])
    tilemap: Grid = [[""] * (cols * local) for _ in range(rows * local)]
    if observer:
        observer({"type": "local_init", "level": "local", "height": rows * local,
                  "width": cols * local, "block_size": local})
    attempts = 0
    for br in range(rows):
        for bc in range(cols):
            check_cancel()
            btype = blocks[br][bc]
            base = [br * local, bc * local]
            relay = _tagged(observer, level="local", br=br, bc=bc, base=base)
            LG = LocalGenerator(btype, local, block_neighbors(blocks, br, bc), rng=rng)
            if relay:
                # Re-announce the block on every attempt so a live view can
                # reset it when a failed attempt is discarded.
                def on_event(event: Dict[str, Any], relay=relay, btype=btype) -> None:
                    if event["type"] == "start":
                        relay({"type": "local_block_start", "block_type": btype, "size": local})
                    relay(event)
                LG.set_observer(on_event, step_delay=step_delay * 0.25)
            ok = LG.collapse()
            attempts += LG.attempts_used
            if not ok:
                issues.append(f"block ({br},{bc}) {btype} invalid after {LG.attempts_used} attempts")
                if strict:
                    raise GenerationError(issues[-1], issues)
            tiles = LG.tile_grid()
            for r in range(local):
                tilemap[base[0] + r][base[1]:base[1] + local] = tiles[r]
            if relay:
                relay({"type": "local_block_complete", "tiles": tiles})
    timings["local"] = time.perf_counter() - t0

    return HAMCMap(width=width, height=height, subgrid=subgrid, local=local, seed=seed,
                   regions=regions, blocks=blocks, tilemap=tilemap, valid=not issues,
                   issues=issues, timings=timings, block_attempts=attempts)
