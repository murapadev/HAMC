# HAMC · Hierarchical Adaptive Model Collapse

<p align="center">
  <a href="https://github.com/murapadev/HAMC"><img alt="Repo" src="https://img.shields.io/badge/repo-murapadev%2Fhamc-24292e?logo=github&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/murapadev/HAMC?color=blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-29%20passing-brightgreen?logo=pytest">
  <a href="https://github.com/murapadev/HAMC/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/murapadev/HAMC?style=social"></a>
</p>

A concise, hierarchical procedural-generation toolkit combining Wave Function Collapse (WFC), adaptive backtracking, and validation to produce multi-level maps (regions → blocks → tiles).

---

## Quick links

- Overview: #overview
- Quickstart: #quickstart
- Live dashboard: #live-dashboard
- Examples: #examples
- Configuration: #configuration
- Tests: #running-tests

## Overview

HAMC is a three-layer procedural content generator:

- Global: assigns regions (e.g., forest, desert, city) across a coarse grid.
- Intermediate: subdivides each global cell into blocks (e.g., grove, sand, residential), handling transitions.
- Local: generates tilemaps inside blocks (e.g., grass, water, sand) with connectivity constraints (rivers, roads).

Core ideas: weighted choices, entropy-based cell selection, constraint propagation, and adaptive backtracking.

## Features

- Hierarchical WFC-style generation across three levels
- Weighted probabilities and Shannon-entropy selection
- Connectivity and path validation for rivers/roads
- Transition zones and special block types (oasis, buildings)
- Adaptive backtracking with failure tracking
- Renderer for per-level visualization
- Live browser dashboard streaming the generation process (SSE, no extra deps)

## Project layout

- main.py — CLI entry point
- dashboard.py — live dashboard server (SSE)
- dashboard_static/ — dashboard frontend (HTML/CSS/JS)
- hamc/
  - config/ — regions, blocks, tiles (weights, compatibility, colours) and tunable constants
  - core/ — cell, entropy, backtracking, validators, generator state
  - generators/ — global, intermediate, local generators
  - visualization/ — rendering utilities
- tests/ — unit tests
- run_tests.py — test runner (pytest + coverage)
- docs/AUDITORIA.md — code audit that motivated the current simplification
- Makefile — `make demo`, `make dashboard`, `make test`

## Quickstart

1) Install

```bash
pip install -r requirements.txt
```

2) Demo (CLI)

```bash
python main.py \
  --width 4 --height 3 --subgrid 2 --local 4 --output output --debug
```

Output includes: output/global_map.png, output/intermediate_map.png, output/final_tilemap.png, and JSON data files.

3) Demo (Python)

```python
from hamc.generators.global_generator import GlobalGenerator
from hamc.generators.intermediate_generator import IntermediateGenerator
from hamc.generators.local_generator import LocalGenerator
from hamc.visualization.renderer import MapRenderer

Wg, Hg, S, L = 4, 3, 2, 4
renderer = MapRenderer(tile_size=20, padding=1)

G = GlobalGenerator(Wg, Hg)
G.initialize(); G.collapse()

I = IntermediateGenerator(G, subgrid_size=S)
I.collapse()

# Stitch local tilemaps
Lh, Lw = len(I.cells), len(I.cells[0])
tilemap = []
for br in range(Lh):
    row_tiles = []
    for bc in range(Lw):
        t = I.cells[br][bc].collapsed_value
        LG = LocalGenerator(t, L)
        LG.collapse()
        row_tiles.append([[cell.collapsed_value for cell in r] for r in LG.cells])
    for r in range(L):
        tilemap.append(sum([b[r] for b in row_tiles], []))

renderer.render_final_map(tilemap, global_size=(Hg, Wg), intermediate_size=(Hg*S, Wg*S)).save('output/final_tilemap.png')
```

## Live dashboard

Watch the WFC process collapse in real time. The dashboard is a tiny stdlib
HTTP server that streams generator events through Server-Sent Events:

```bash
python dashboard.py --open          # serves http://127.0.0.1:8765
# or
make dashboard PORT=9000
```

The browser UI renders three synchronized canvases (global → intermediate →
local) and pulses each cell as it collapses. The right-hand log shows
collapses, conflicts, and backtracks with timestamps. Adjust grid sizes,
step delay and seed from the side panel.

Hook into the same event stream programmatically by registering a callback:

```python
def observer(event):
    print(event["type"], event.get("pos"), event.get("value"))

G = GlobalGenerator(4, 3)
G.set_observer(observer, step_delay=0.05)
G.collapse()
```

## Configuration

Everything is plain Python in hamc/config/:

- region_config.py, block_config.py, tile_config.py — option weights, adjacency
  graphs (made symmetric at import time) and colours for each level.
- advanced_config.py — a few tunable probabilities for local generation and
  validation, exposed through `get_generator_config()` / `get_validator_config()`.

There is no profile system or config file loader; edit the values directly.

## Running tests

```bash
python run_tests.py      # or: make test, or: pytest tests
```

Tests cover entropy, compatibility, path validation, propagation/backtracking,
seed reproducibility, the audit regressions and the dashboard request limits.

## Contributing

- Open issues for bugs or enhancements
- Fork and create a descriptive PR
- Follow existing test patterns; run run_tests.py

## License

See LICENSE in the repository.

## Contact

Maintained by @murapadev. Questions or ideas? Open an issue or reach out via GitHub.
