import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..core.cell import Cell
from ..core.generator_state import GeneratorState, GeneratorStatus

Observer = Callable[[Dict[str, Any]], None]
Snapshot = List[List[Tuple[Dict[str, float], Optional[str]]]]

_DIRECTIONS = (("top", -1, 0), ("bottom", 1, 0), ("left", 0, -1), ("right", 0, 1))


class BaseGenerator(ABC):
    """Wave function collapse over a 2D grid of :class:`Cell`.

    Subclasses provide the initial cell domains (``initialize``), the
    adjacency table (``compatibility``) and the final checks (``validate``).
    The solver keeps the grid arc-consistent: whenever a cell loses options,
    each neighbour drops every value that no longer has a compatible partner.
    A contradiction rolls back the most recent decision, bans the value that
    caused it and tries again (chronological backtracking).
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

        # Domains seeded before collapse (forced values, block layouts) must
        # constrain their neighbours before the first decision.
        everything = [(r, c) for r in range(self.height) for c in range(self.width)]
        if not self._propagate_many(everything):
            self.logger.debug("Initial constraints are contradictory")
            return False

        decisions: List[Tuple[Snapshot, Tuple[int, int], str]] = []
        while True:
            target = self._find_min_entropy_cell()
            if target is None:
                return True

            r, c = target
            entropy = self.cells[r][c].entropy()
            snapshot = self._snapshot()
            before = self._counts() if self._observer else None

            self.state.increment_stat("collapses")
            value = self.cells[r][c].collapse(context={"neighbor_values": self._neighbour_values(r, c)},
                                              rng=self.rng)
            if value != "ERROR" and self.propagate(r, c):
                decisions.append((snapshot, (r, c), value))
                self.state.update_entropy(entropy)
                self._emit("collapse", pos=[r, c], value=value, entropy=entropy,
                           propagated=self._patches(before, (r, c)) if before else [])
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
                self._restore(snapshot)
                cell = self.cells[r][c]
                cell.possible.pop(value, None)
                if cell.possible and self.propagate(r, c):
                    self._emit("backtrack", pos=[r, c], grid=self._grid_snapshot())
                    break
                if not decisions:
                    return False
                snapshot, (r, c), value = decisions.pop()

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
                        if not table.get(v, set()).isdisjoint(options)}
                if not kept:
                    return False
                if len(kept) != len(neighbour.possible):
                    neighbour.possible = kept
                    queue.append((nr, nc))
        return True

    # -- cell selection ------------------------------------------------------

    def _find_min_entropy_cell(self) -> Optional[Tuple[int, int]]:
        """Lowest-entropy open cell; ties prefer cells next to collapsed ones
        (organic growth) and are otherwise broken at random."""
        min_entropy = float("inf")
        contenders: List[Tuple[int, int]] = []
        for r in range(self.height):
            for c in range(self.width):
                cell = self.cells[r][c]
                if cell.collapsed_value is not None:
                    continue
                entropy = cell.entropy()
                if entropy < min_entropy - 1e-9:
                    min_entropy = entropy
                    contenders = [(r, c)]
                elif abs(entropy - min_entropy) <= 1e-9:
                    contenders.append((r, c))
        if not contenders:
            return None
        adjacent = [pos for pos in contenders if self._has_collapsed_neighbor(*pos)]
        return self.rng.choice(adjacent or contenders)

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

    def _counts(self) -> Dict[Tuple[int, int], int]:
        return {(r, c): len(self.cells[r][c].possible)
                for r in range(self.height) for c in range(self.width)
                if self.cells[r][c].collapsed_value is None}

    def _patches(self, before: Dict[Tuple[int, int], int], origin: Tuple[int, int]) -> List[List[Any]]:
        """Cells whose option count changed since ``before`` (dashboard diff)."""
        patches: List[List[Any]] = []
        for (r, c), n_before in before.items():
            if (r, c) == origin:
                continue
            cell = self.cells[r][c]
            if len(cell.possible) != n_before:
                patches.append([r, c, len(cell.possible)])
        return patches

    def _snapshot(self) -> Snapshot:
        return [[(dict(cell.possible), cell.collapsed_value) for cell in row] for row in self.cells]

    def _restore(self, snapshot: Snapshot) -> None:
        for r, row in enumerate(snapshot):
            for c, (possible, collapsed) in enumerate(row):
                self.cells[r][c].possible = dict(possible)
                self.cells[r][c].collapsed_value = collapsed

    def tile_grid(self) -> List[List[Optional[str]]]:
        """Collapsed values as a plain 2D list."""
        return [[cell.collapsed_value for cell in row] for row in self.cells]
