# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/) while the API is still `0.x`.

## [0.2.0] - 2026-10-30

Polishing pass on top of the simplification below: correctness, packaging,
performance, a renderer and documentation.

### Added
- `hamc.generate_map()` pipeline returning a `HAMCMap` (layers, seed,
  validity, issues, timings), with `strict` mode, an observer hook tagged by
  layer and cancellation.
- Customisation: `region_weights`, `block_weights`, `tile_weights`,
  `constrain_regions(row, col)` and `constrain_blocks(row, col, region)`.
- `park` block in city regions.
- Procedural renderer `hamc.render_map()` with `topdown`, `iso` and `flat`
  styles; no image assets.
- `hamc` CLI: `generate`, `render` (`--size`, `--style`, `--tile-size`) and
  `dashboard`; also `python -m hamc`.
- Dashboard panel with the rendered map (`GET /api/render`).
- `examples/` (nine runnable scripts), `benchmarks/bench.py`,
  `docs/make_images.py`, GitHub Actions CI (ruff + pytest on 3.11–3.13).

### Changed
- Solver rewritten around arc-consistent propagation from the start
  (including pre-seeded layouts), chronological backtracking with an undo
  trail and an entropy heap with lazy invalidation. A 64×64 grid went from
  27.8 s to 0.13 s (see README, Performance).
- One `random.Random(seed)` per run, injected into every layer, replaces the
  global `random`/`numpy` state; runs are reproducible across processes and
  safe to run concurrently.
- Transition blocks are options on region borders instead of forced cells.
- River and road blocks lay their channel towards neighbouring rivers/roads;
  a single validator checks that all connected sides are reachable.
- Packaging moved from a broken `setup.py` to `pyproject.toml`; the dashboard
  and its static files are part of the `hamc` package. NumPy is no longer a
  dependency.
- README rewritten in English.

### Fixed
- Commercial and industrial blocks failed validation almost always and roads
  on larger blocks; `main.py` rendered them anyway. Invalid blocks are now
  regenerated (up to 25 attempts) and a map with an invalid block raises
  `GenerationError` (or is flagged `valid=False` with `strict=False`).
- Backtracking crashed on a level-name mismatch and never restored state.
- `road` blocks could not be adjacent to each other.
- Grids too small for every region type (or blocks under 3×3) now fail with
  a clear `ValueError` instead of silently missing regions.

### Removed
- `main.py`, `dashboard.py` at the repository root (now `hamc.cli`,
  `hamc.dashboard`), `setup.py`, `requirements.txt`, `run_tests.py`,
  `BacktrackManager`, and the unused history/metadata machinery in `Cell`.

## [0.1.0] - 2026-10-29

Simplification merged in #1, following the audit in `docs/AUDITORIA.md`.

### Added
- Live dashboard streaming generation events over Server-Sent Events, with
  concurrency, size and CORS limits.
- Richer tile palette and block layouts.

### Fixed
- Correctness fixes from the audit: commercial-block validation typo,
  asymmetric adjacency tables, `collapse()` wiping pre-seeded cells, silent
  failures in weighted choice.

### Removed
- Unused "enterprise" scaffolding (DI container, plugins, factories,
  strategies, parallel generator, adaptive entropy analyzers) and unused
  dependencies.
