import heapq
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..core.cell import Cell
from ..core.generator_state import GeneratorState, GeneratorStatus

Observer = Callable[[Dict[str, Any]], None]

_EMPTY: Set[str] = frozenset()
_DIRECTIONS = (("top", -1, 0), ("bottom", 1, 0), ("left", 0, -1), ("right", 0, 1))


class BaseGenerator(ABC):
    """Wave function collapse over a 2D grid of :class:`Cell`.

    Subclasses provide the initial cell domains (``initialize``), the
    adjacency table (``compatibility``) and the final checks (``validate``).
    The solver keeps the grid arc-consistent: whenever a cell loses options,
    each neighbour drops every value that no longer has a compatible partner.
    A contradiction rolls back the most recent decision, bans the value that
    caused it and tries again (chronological backtracking).

    Cost per step is proportional to what changed, not to the grid: the next
    cell comes from an entropy heap with lazy invalidation, and rollbacks
    replay an undo trail instead of restoring full-grid snapshots.
    """

    #: Upper bound on rollbacks per collapse() call; ``None`` scales with the grid.
    max_backtracks: Optional[int] = None

    def __init__(self, width: int, height: int, name: str,
                 rng: Optional[random.Random] = None):
        self.width = width
        self.height = height
        self.cells: List[List[Cell]] = []
        self.state = GeneratorState(name)
        self.logger = logging.getLogger(name)
        self.name = name
        self.backtracks = 0
        # Every random draw goes through this instance, so a seeded Random
        # reproduces a run exactly and concurrent runs don't interfere.
        self.rng = rng or random.Random()
        self._observer: Optional[Observer] = None
        self.step_delay: float = 0.0
        # Undo trail: (row, col, previous domain, previous collapsed value).
        # Domains are replaced, never mutated, so keeping references is safe.
        self._trail: List[Tuple[int, int, Dict[str, float], Optional[str]]] = []
        # Entropy heap entries: (entropy, not-next-to-collapsed, tie-break, r, c, stamp).
        self._heap: List[Tuple[float, int, float, int, int, int]] = []
        self._stamp: Optional[List[List[int]]] = None

    # -- hooks ---------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> bool:
        """Fill ``self.cells`` with the initial domains."""

    @abstractmethod
    def validate(self) -> bool:
        """Check the fully collapsed grid."""

    @abstractmethod
    def compatibility(self) -> Dict[str, Set[str]]:
        """Symmetric adjacency table: value -> values allowed next to it."""

    # -- observer ------------------------------------------------------------

    def set_observer(self, observer: Optional[Observer], step_delay: float = 0.0) -> None:
        """Register a callback that receives generation events.

        Events are dicts with a ``type`` key (``start``, ``init``, ``collapse``,
        ``conflict``, ``backtrack``, ``complete``, ``fail``) plus payload.
        ``step_delay`` sleeps after each collapse, to animate the process.
        """
        self._observer = observer
        self.step_delay = max(0.0, float(step_delay))

    def _emit(self, event_type: str, **payload: Any) -> None:
        if not self._observer:
            return
        try:
            self._observer({"type": event_type, "generator": self.name, **payload})
        except Exception as e:  # an observer must never break generation
            self.logger.debug("Observer raised, ignoring: %s", e)

    def _grid_snapshot(self) -> List[List[Dict[str, Any]]]:
        return [[{"v": cell.collapsed_value, "n": len(cell.possible)} for cell in row]
                for row in self.cells]

    # -- main loop -----------------------------------------------------------

    def collapse(self) -> bool:
        """Collapse every cell, then validate. Returns ``True`` on success."""
        self.state.update_status(GeneratorStatus.IN_PROGRESS)
        self._emit("start", width=self.width, height=self.height)
        try:
            # Subclasses may seed ``self.cells`` before collapsing; only build
            # the grid when nothing is there yet so those seeds survive.
            if not self.cells and not self.initialize():
                return self._fail("initialize")
            self._emit("init", grid=self._grid_snapshot())
            if not self._run_collapse():
                return self._fail("collapse")
            if not self.validate():
                return self._fail("validate")
            self.state.update_status(GeneratorStatus.COMPLETED)
            self._emit("complete", grid=self._grid_snapshot())
            return True
        except Exception as e:
            self.logger.exception("Error during collapse")
            return self._fail(str(e))

    def _fail(self, reason: str) -> bool:
        self.state.update_status(GeneratorStatus.FAILED, reason)
        self._emit("fail", reason=reason)
        return False

    def _run_collapse(self) -> bool:
        self.backtracks = 0
        limit = self.max_backtracks or max(50, self.width * self.height)
        self._trail = []
        self._heap = []
        self._stamp = [[0] * self.width for _ in range(self.height)]
        for r in range(self.height):
            for c in range(self.width):
                self._push(r, c)

        # Domains seeded before collapse (forced values, block layouts) must
        # constrain their neighbours before the first decision.
        everything = [(r, c) for r in range(self.height) for c in range(self.width)]
        if not self._propagate_many(everything):
            self.logger.debug("Initial constraints are contradictory")
            return False

        decisions: List[Tuple[int, Tuple[int, int], str]] = []
        while True:
            target = self._find_min_entropy_cell()
            if target is None:
                return True

            r, c = target
            cell = self.cells[r][c]
            entropy = cell.entropy()
            mark = len(self._trail)

            self.state.increment_stat("collapses")
            self._record(r, c)
            value = cell.collapse(context={"neighbor_values": self._neighbour_values(r, c)},
                                  rng=self.rng)
            self._touch(r, c)
            if value != "ERROR" and self.propagate(r, c):
                decisions.append((mark, (r, c), value))
                self.state.update_entropy(entropy)
                if self._observer:
                    self._emit("collapse", pos=[r, c], value=value, entropy=entropy,
                               propagated=self._patches(mark, (r, c)))
                if self.step_delay > 0.0:
                    time.sleep(self.step_delay)
                continue

            self._emit("conflict", pos=[r, c], value=value)
            # Undo the failed decision; if banning its value empties the cell
            # (or contradicts the rest), undo the previous decision as well.
            while True:
                self.backtracks += 1
                self.state.increment_stat("backtracks")
                if self.backtracks > limit:
                    self.logger.debug("Backtrack limit (%d) reached", limit)
                    return False
                self._undo_to(mark)
                remaining = {k: w for k, w in self.cells[r][c].possible.items() if k != value}
                if remaining:
                    self._set_possible(r, c, remaining)
                    if self.propagate(r, c):
                        if self._observer:
                            self._emit("backtrack", pos=[r, c], grid=self._grid_snapshot())
                        break
                if not decisions:
                    return False
                mark, (r, c), value = decisions.pop()

    # -- constraint propagation ---------------------------------------------

    def propagate(self, row: int, col: int) -> bool:
        """Make the grid arc-consistent after ``(row, col)`` lost options."""
        return self._propagate_many([(row, col)])

    def _propagate_many(self, sources: List[Tuple[int, int]]) -> bool:
        table = self.compatibility()
        queue = deque(sources)
        while queue:
            r, c = queue.popleft()
            options = self.cells[r][c].possible
            for nr, nc in self.get_neighbors(r, c):
                neighbour = self.cells[nr][nc]
                kept = {v: w for v, w in neighbour.possible.items()
                        if not table.get(v, _EMPTY).isdisjoint(options)}
                if not kept:
                    return False
                if len(kept) != len(neighbour.possible):
                    self._set_possible(nr, nc, kept)
                    queue.append((nr, nc))
        return True

    # -- undo trail and entropy heap ----------------------------------------

    def _record(self, r: int, c: int) -> None:
        cell = self.cells[r][c]
        self._trail.append((r, c, cell.possible, cell.collapsed_value))

    def _set_possible(self, r: int, c: int, possible: Dict[str, float]) -> None:
        self._record(r, c)
        self.cells[r][c].possible = possible
        if self._stamp is not None:
            self._push(r, c)

    def _touch(self, r: int, c: int) -> None:
        """Re-rank the open neighbours of a cell whose collapsed state changed."""
        for nr, nc in self.get_neighbors(r, c):
            if self.cells[nr][nc].collapsed_value is None:
                self._push(nr, nc)

    def _undo_to(self, mark: int) -> None:
        trail = self._trail
        while len(trail) > mark:
            r, c, possible, collapsed = trail.pop()
            cell = self.cells[r][c]
            was_collapsed = cell.collapsed_value is not None
            cell.possible = possible
            cell.collapsed_value = collapsed
            if collapsed is None:
                self._push(r, c)
            if was_collapsed != (collapsed is not None):
                self._touch(r, c)

    def _push(self, r: int, c: int) -> None:
        """(Re)insert a cell in the entropy heap; older entries go stale."""
        cell = self.cells[r][c]
        if cell.collapsed_value is not None:
            return
        self._stamp[r][c] += 1
        heapq.heappush(self._heap, (round(cell.entropy(), 9),
                                    0 if self._has_collapsed_neighbor(r, c) else 1,
                                    self.rng.random(), r, c, self._stamp[r][c]))

    def _find_min_entropy_cell(self) -> Optional[Tuple[int, int]]:
        """Lowest-entropy open cell. Ties prefer cells next to a collapsed one
        (organic growth) and are otherwise broken at random. Entries whose
        cell changed since they were pushed are skipped (lazy invalidation)."""
        heap = self._heap
        while heap:
            _, _, _, r, c, stamp = heapq.heappop(heap)
            if stamp == self._stamp[r][c] and self.cells[r][c].collapsed_value is None:
                return r, c
        return None

    def _has_collapsed_neighbor(self, r: int, c: int) -> bool:
        return any(self.cells[nr][nc].collapsed_value is not None
                   for nr, nc in self.get_neighbors(r, c))

    def _neighbour_values(self, r: int, c: int) -> Dict[str, str]:
        """Collapsed neighbours keyed by direction (biases Cell.collapse)."""
        out: Dict[str, str] = {}
        for direction, dr, dc in _DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                v = self.cells[nr][nc].collapsed_value
                if v:
                    out[direction] = v
        return out

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        return [(r + dr, c + dc) for _, dr, dc in _DIRECTIONS
                if 0 <= r + dr < self.height and 0 <= c + dc < self.width]

    # -- state helpers -------------------------------------------------------

    def _patches(self, mark: int, origin: Tuple[int, int]) -> List[List[Any]]:
        """Cells whose domain changed since ``mark`` (dashboard diff)."""
        seen = {origin}
        patches: List[List[Any]] = []
        for r, c, _, _ in self._trail[mark:]:
            if (r, c) not in seen:
                seen.add((r, c))
                patches.append([r, c, len(self.cells[r][c].possible)])
        return patches

    def tile_grid(self) -> List[List[Optional[str]]]:
        """Collapsed values as a plain 2D list."""
        return [[cell.collapsed_value for cell in row] for row in self.cells]
