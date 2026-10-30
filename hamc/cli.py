"""Command line interface: ``hamc generate`` and ``hamc dashboard``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

from . import __version__
from .pipeline import GenerationError, HAMCMap, generate_map


def _grid_options() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("map size")
    g.add_argument("--width", type=int, default=4, help="region grid width (default: 4)")
    g.add_argument("--height", type=int, default=3, help="region grid height (default: 3)")
    g.add_argument("--subgrid", type=int, default=2, help="blocks per region side (default: 2)")
    g.add_argument("--local", type=int, default=4, help="tiles per block side (default: 4)")
    g.add_argument("--seed", type=int, default=None, help="seed for a reproducible map")
    p.add_argument("--debug", action="store_true", help="verbose logging")
    return p


def _generate(ns: argparse.Namespace) -> HAMCMap:
    return generate_map(ns.width, ns.height, ns.subgrid, ns.local, seed=ns.seed)


def write_layers(hmap: HAMCMap, out_dir: str) -> Dict[str, str]:
    """Write the three layers as flat-colour PNGs plus ``map.json``."""
    from .visualization.renderer import MapRenderer

    os.makedirs(out_dir, exist_ok=True)
    renderer = MapRenderer(tile_size=20, padding=1)
    images = {
        "global_map": renderer.render_global_map(hmap.regions),
        "intermediate_map": renderer.render_intermediate_map(hmap.blocks),
        "final_tilemap": renderer.render_final_map(
            hmap.tilemap, global_size=(hmap.height, hmap.width),
            intermediate_size=(hmap.height * hmap.subgrid, hmap.width * hmap.subgrid)),
        "final_tilemap_clean": renderer.render_complete_tilemap(hmap.tilemap),
    }
    paths: Dict[str, str] = {}
    for name, img in images.items():
        paths[name] = os.path.join(out_dir, f"{name}.png")
        img.save(paths[name])
    paths["map"] = os.path.join(out_dir, "map.json")
    hmap.save_json(paths["map"])
    return paths


def cmd_generate(ns: argparse.Namespace) -> int:
    hmap = _generate(ns)
    if ns.json:
        json.dump(hmap.to_dict(), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    paths = write_layers(hmap, ns.output)
    rows, cols = len(hmap.tilemap), len(hmap.tilemap[0])
    print(f"seed {hmap.seed}: {cols}x{rows} tiles in {sum(hmap.timings.values()):.2f}s")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def cmd_dashboard(ns: argparse.Namespace) -> int:
    from .dashboard import serve
    return serve(ns.host, ns.port, ns.open)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamc", description="Hierarchical Adaptive Model Collapse: procedural maps in three layers.")
    parser.add_argument("--version", action="version", version=f"hamc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    grid = _grid_options()

    p = sub.add_parser("generate", parents=[grid], help="generate a map and write its layers",
                       description="Generate a map; write the region, block and tile layers "
                                   "as PNG plus map.json.")
    p.add_argument("-o", "--output", default="output", help="output directory (default: output)")
    p.add_argument("--json", action="store_true", help="print the map as JSON to stdout instead")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("dashboard", help="serve the live dashboard",
                       description="Serve the live dashboard (Server-Sent Events, stdlib only).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open a browser tab")
    p.set_defaults(func=cmd_dashboard, debug=False)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    ns = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if ns.debug else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return ns.func(ns)
    except (GenerationError, ValueError) as e:
        print(f"hamc: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
