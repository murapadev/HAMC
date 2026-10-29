from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Callable, Any
import time
import logging
from ..core.cell import Cell
from ..core.generator_state import GeneratorState, GeneratorStatus
from ..core.backtrack_manager import BacktrackManager

Observer = Callable[[Dict[str, Any]], None]


class BaseGenerator(ABC):
    """Base class for all wave function collapse generators."""

    def __init__(self, width: int, height: int, name: str):
        self.width = width
        self.height = height
        self.cells: List[List[Cell]] = []
        self.state = GeneratorState(name)
        self.backtrack_manager = BacktrackManager()
        self.logger = logging.getLogger(name)
        self.name = name
        self._observer: Optional[Observer] = None
        self.step_delay: float = 0.0

        # Configure logger if not already configured. ``hasHandlers`` also
        # checks ancestors, so a logging-configured app (e.g. the dashboard)
        # doesn't get noisy duplicate output.
        if not self.logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def set_observer(self, observer: Optional[Observer], step_delay: float = 0.0) -> None:
        """Register a callback to receive generation events for live streaming.

        Events are dicts with at least a ``type`` key plus payload fields.
        Setting observer to ``None`` disables emission. ``step_delay`` is the
        sleep in seconds after each successful cell collapse — useful for
        animating the process in real time.
        """
        self._observer = observer
        self.step_delay = max(0.0, float(step_delay))

    def _emit(self, event_type: str, **payload: Any) -> None:
        if not self._observer:
            return
        try:
            event = {"type": event_type, "generator": self.name, **payload}
            self._observer(event)
        except Exception as e:
            self.logger.debug(f"Observer raised, ignoring: {e}")

    def _grid_snapshot(self) -> List[List[Dict[str, Any]]]:
        """Return a compact per-cell snapshot used by the dashboard."""
        snap: List[List[Dict[str, Any]]] = []
        for row in self.cells:
            row_snap = []
            for cell in row:
                row_snap.append({
                    "v": cell.collapsed_value,
                    "n": len(cell.possible),
                })
            snap.append(row_snap)
        return snap

    def collapse(self) -> bool:
        """Template method for wave function collapse."""
        self.state.update_status(GeneratorStatus.IN_PROGRESS)
        self._emit("start", width=self.width, height=self.height)

        try:
            # Only initialize when the grid hasn't been set up yet. Subclasses
            # (and the forced-region path in GlobalGenerator) seed ``self.cells``
            # before calling ``collapse``; re-initializing here would wipe that
            # seeding — the root cause of regions silently going missing and of
            # the local "strategic" pre-collapse being thrown away.
            if not self.cells:
                if not self.initialize():
                    self.state.update_status(GeneratorStatus.FAILED, "Initialization failed")
                    self._emit("fail", reason="initialize")
                    return False

            self._emit("init", grid=self._grid_snapshot())

            if not self._run_collapse():
                self.state.update_status(GeneratorStatus.FAILED, "Collapse failed")
                self._emit("fail", reason="collapse")
                return False

            if not self.validate():
                self.state.update_status(GeneratorStatus.FAILED, "Validation failed")
                self._emit("fail", reason="validate")
                return False

            self.state.update_status(GeneratorStatus.COMPLETED)
            self._emit("complete", grid=self._grid_snapshot())
            return True

        except Exception as e:
            self.state.logger.error(f"Error during collapse: {str(e)}")
            self.state.update_status(GeneratorStatus.FAILED, str(e))
            self._emit("fail", reason=str(e))
            return False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the generator."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate the final state."""
        pass

    def _run_collapse(self) -> bool:
        """Execute wave function collapse with improved backtracking."""
        self.backtrack_manager.clear()

        # Stream protocol: we ship a full grid snapshot on init/backtrack/
        # complete and per-collapse *patches* describing the collapsed cell
        # plus any neighbours whose option-count changed due to propagation.
        # The client maintains its own grid replica from those patches, which
        # keeps payloads small and lets the UI animate propagation.

        while True:
            target = self._find_min_entropy_cell()

            if not target:
                return True  # All cells collapsed successfully

            r, c = target
            current_entropy = self.cells[r][c].entropy()

            if current_entropy == float('inf'):
                point = self.backtrack_manager.pop()
                if not point:
                    return False  # No more backtrack points

                self.state.update_status(GeneratorStatus.BACKTRACKING)
                self._restore(point.state_snapshot)
                self._emit(
                    "backtrack",
                    pos=list(point.position),
                    grid=self._grid_snapshot(),
                )
                continue

            # Take snapshot before collapsing
            snapshot = self._snapshot()
            self.backtrack_manager.push(
                self.__class__.__name__.lower(),
                (r, c),
                snapshot,
                current_entropy,
                {"failed_values": self.backtrack_manager.get_failed_values(
                    self.__class__.__name__.lower(), (r, c)
                )}
            )

            # Snapshot option-counts of uncollapsed cells so we can diff
            # against them after propagation finishes.
            pre_counts: Dict[Tuple[int, int], int] = {}
            for rr in range(self.height):
                for cc in range(self.width):
                    cell = self.cells[rr][cc]
                    if cell.collapsed_value is None:
                        pre_counts[(rr, cc)] = len(cell.possible)

            self.state.increment_stat("collapses")
            ctx = {"neighbor_values": self._neighbour_values(r, c)}
            value = self.cells[r][c].collapse(context=ctx)

            if not value or not self.propagate(r, c):
                self.backtrack_manager.record_failure(
                    self.__class__.__name__.lower(),
                    (r, c),
                    value or "propagation_failure"
                )
                self._emit("conflict", pos=[r, c], value=value)
                continue

            # Build a list of patches: cells (other than the collapsed one)
            # whose option-count or collapsed_value changed by propagation.
            patches: List[List[Any]] = []
            for (rr, cc), n_before in pre_counts.items():
                if rr == r and cc == c:
                    continue
                cell = self.cells[rr][cc]
                if cell.collapsed_value is not None:
                    patches.append([rr, cc, 0, cell.collapsed_value])
                elif len(cell.possible) != n_before:
                    patches.append([rr, cc, len(cell.possible)])

            self.state.update_entropy(current_entropy)
            self._emit(
                "collapse",
                pos=[r, c],
                value=value,
                entropy=current_entropy,
                propagated=patches,
            )
            if self.step_delay > 0.0:
                time.sleep(self.step_delay)

    def _find_min_entropy_cell(self) -> Optional[Tuple[int, int]]:
        """Find a cell to collapse next.

        Strategy:
        1. Drop values previously marked as failed at each position.
        2. Track the minimum *positive* entropy across the grid.
        3. Among all cells at that minimum, prefer those adjacent to at least
           one already-collapsed cell — this produces organic growth from
           the initial seed rather than scattered noise. Among the remaining
           candidates, pick uniformly at random so ties don't produce
           scanline patterns.
        """
        import random as _random

        min_entropy = float('inf')
        contenders: List[Tuple[int, int]] = []

        for r in range(self.height):
            for c in range(self.width):
                cell = self.cells[r][c]
                if cell.collapsed_value is not None:
                    continue

                failed_values = self.backtrack_manager.get_failed_values(
                    self.__class__.__name__.lower(),
                    (r, c)
                )
                if failed_values:
                    valid_possible = {k: v for k, v in cell.possible.items()
                                      if k not in failed_values}
                    if valid_possible:
                        cell.possible = valid_possible

                entropy = cell.entropy()
                if entropy < 0:
                    continue
                if entropy < min_entropy - 1e-9:
                    min_entropy = entropy
                    contenders = [(r, c)]
                elif abs(entropy - min_entropy) <= 1e-9:
                    contenders.append((r, c))

        if not contenders:
            return None

        # Among ties, favour cells with a collapsed neighbour so the wave
        # grows continuously instead of jumping around the grid.
        adjacent = [pos for pos in contenders if self._has_collapsed_neighbor(*pos)]
        pool = adjacent if adjacent else contenders
        return _random.choice(pool)

    def _has_collapsed_neighbor(self, r: int, c: int) -> bool:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if self.cells[nr][nc].collapsed_value is not None:
                    return True
        return False

    def _neighbour_values(self, r: int, c: int) -> Dict[str, str]:
        """Return collapsed-neighbour values keyed by direction.

        Used to bias Cell.collapse toward matching its already-decided
        neighbours, which produces tighter visual clusters.
        """
        out: Dict[str, str] = {}
        for direction, (dr, dc) in (
            ("top", (-1, 0)), ("bottom", (1, 0)),
            ("left", (0, -1)), ("right", (0, 1)),
        ):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                v = self.cells[nr][nc].collapsed_value
                if v:
                    out[direction] = v
        return out

    @abstractmethod
    def propagate(self, row: int, col: int) -> bool:
        """Propagate constraints from a cell."""
        pass

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Get valid neighbors for a cell."""
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbors.append((nr, nc))
        return neighbors

    def _snapshot(self) -> List[List[Tuple[Dict[str, float], str]]]:
        """Take a snapshot of current state for backtracking."""
        snap = []
        for r in range(self.height):
            row_snap = []
            for c in range(self.width):
                cell = self.cells[r][c]
                row_snap.append((dict(cell.possible), cell.collapsed_value))
            snap.append(row_snap)
        return snap

    def _restore(self, snapshot: List[List[Tuple[Dict[str, float], str]]]) -> None:
        """Restore state from a snapshot."""
        for r in range(self.height):
            for c in range(self.width):
                possible, collapsed = snapshot[r][c]
                self.cells[r][c].possible = dict(possible)
                self.cells[r][c].collapsed_value = collapsed
