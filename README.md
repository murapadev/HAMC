# HAMC · Hierarchical Adaptive Model Collapse

<p align="center">
  <a href="https://github.com/murapadev/HAMC"><img alt="Repo" src="https://img.shields.io/badge/repo-murapadev%2Fhamc-24292e?logo=github&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/murapadev/HAMC?color=blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-68%20passing-brightgreen?logo=pytest">
  <a href="https://github.com/murapadev/HAMC/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/murapadev/HAMC?style=social"></a>
</p>

A concise, hierarchical procedural-generation toolkit combining Wave Function Collapse (WFC), adaptive backtracking, and validation to produce multi-level maps (regions → blocks → tiles).

---

## Quick links

- Overview: #overview
- Quickstart: #quickstart
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

## Project layout

- main.py — CLI entry point
- hamc/
  - config/ — configuration files
  - core/ — cell, entropy, backtracking, validators
  - generators/ — global, intermediate, local generators
  - visualization/ — rendering utilities
- tests/ — unit tests
- run_tests.py — test runner

For a full tree see the original README in the repo.

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

## Configuration

All constants and profiles live in hamc/config/. The project supports profile-based JSON configs (development, production, testing) and a runtime loader at hamc.config.advanced_config.

## Running tests

```bash
python run_tests.py
```

Tests cover entropy, compatibility, path validation, and propagation/backtracking behavior.

## Contributing

- Open issues for bugs or enhancements
- Fork and create a descriptive PR
- Follow existing test patterns; run run_tests.py

## License

See LICENSE in the repository.

## Contact

Maintained by @murapadev. Questions or ideas? Open an issue or reach out via GitHub.
