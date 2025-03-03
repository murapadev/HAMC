#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import logging
import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from hamc.generators.global_generator import GlobalGenerator
from hamc.generators.intermediate_generator import IntermediateGenerator
from hamc.generators.local_generator import LocalGenerator
from hamc.visualization.renderer import MapRenderer
from hamc.core.generator_state import GeneratorStatus

class HAMCMapGenerator:
    """Main class for Hierarchical Adaptive Model Collapse map generation."""

    def __init__(self, 
                 global_size: Tuple[int, int],
                 subgrid_size: int = 3,
                 local_size: int = 5,
                 output_dir: str = "output",
                 seed: Optional[int] = None,
                 log_level: int = logging.INFO):
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
        
        # Initialize random seed
        if seed is not None:
            random.seed(seed)
            self.logger.info(f"Random seed set to: {seed}")
        
        self.global_size = global_size
        self.subgrid_size = subgrid_size
        self.local_size = local_size
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize generators
        self.global_gen = GlobalGenerator(*global_size)
        self.intermediate_gen = None
        self.complete_tilemap = None
        
        # Initialize renderer
        self.renderer = MapRenderer(tile_size=20)
        
        self.logger.info(f"Initialized HAMC generator with global size {global_size}")

    def generate(self) -> bool:
        """Generate complete map through all levels with detailed progress tracking."""
        try:
            self.logger.info("Starting map generation...")
            start_time = datetime.now()

            # Generate global level
            self.logger.info("Generating global level...")
            if not self._generate_global_level():
                self.logger.error("Global level generation failed")
                return False

            # Generate intermediate level
            self.logger.info("Generating intermediate level...")
            if not self._generate_intermediate_level():
                self.logger.error("Intermediate level generation failed")
                return False

            # Generate local level
            self.logger.info("Generating local level...")
            if not self._generate_local_level():
                self.logger.error("Local level generation failed")
                return False

            # Save generation metrics
            end_time = datetime.now()
            metrics = {
                "global_metrics": self.global_gen.state.get_summary(),
                "intermediate_metrics": self.intermediate_gen.state.get_summary() if self.intermediate_gen else None,
                "total_time": str(end_time - start_time)
            }
            
            metrics_file = self.run_dir / "generation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            self.logger.info(f"Generation completed successfully in {end_time - start_time}")
            return True

        except Exception as e:
            self.logger.exception("Unexpected error during generation")
            return False

    def _generate_global_level(self) -> bool:
        """Generate and validate global level."""
        try:
            self.global_gen.initialize()
            success = self.global_gen.collapse()
            
            if success:
                # Save visualization
                global_map = [[cell.collapsed_value for cell in row] 
                            for row in self.global_gen.cells]
                img = self.renderer.render_global_map(global_map)
                img.save(self.run_dir / "global_level.png")
                
                # Save map data
                with open(self.run_dir / "global_map.json", 'w') as f:
                    json.dump(global_map, f, indent=2)
                
                # Log map preview
                self.logger.info("\nGlobal level map:")
                for row in global_map[:min(3, len(global_map))]:
                    self.logger.info(" | ".join(cell or "None" for cell in row))
                if len(global_map) > 3:
                    self.logger.info("...")
            
            return success

        except Exception as e:
            self.logger.exception("Error in global level generation")
            return False

    def _generate_intermediate_level(self) -> bool:
        """Generate and validate intermediate level."""
        try:
            self.intermediate_gen = IntermediateGenerator(
                self.global_gen,
                self.subgrid_size
            )
            success = self.intermediate_gen.collapse()
            
            if success:
                # Save visualization
                intermediate_map = [[cell.collapsed_value for cell in row] 
                                  for row in self.intermediate_gen.cells]
                img = self.renderer.render_intermediate_map(intermediate_map)
                img.save(self.run_dir / "intermediate_level.png")
                
                # Save map data
                with open(self.run_dir / "intermediate_map.json", 'w') as f:
                    json.dump(intermediate_map, f, indent=2)
                
                # Log preview
                self.logger.info("\nIntermediate level preview (first 3x3):")
                for row in intermediate_map[:min(3, len(intermediate_map))]:
                    self.logger.info(" | ".join(cell or "None" for cell in row[:3]))
            
            return success

        except Exception as e:
            self.logger.exception("Error in intermediate level generation")
            return False

    def _generate_local_level(self) -> bool:
        """Generate and validate local level tiles."""
        try:
            if not self.intermediate_gen:
                return False
                
            total_height = self.intermediate_gen.height * self.local_size
            total_width = self.intermediate_gen.width * self.local_size
            self.complete_tilemap = [["Empty" for _ in range(total_width)]
                                   for _ in range(total_height)]
            
            # Create directory for local blocks
            local_dir = self.run_dir / "local_blocks"
            local_dir.mkdir(exist_ok=True)
            
            # Track progress
            total_blocks = self.intermediate_gen.height * self.intermediate_gen.width
            completed_blocks = 0
            
            # Generate local tilemaps
            for br in range(self.intermediate_gen.height):
                for bc in range(self.intermediate_gen.width):
                    block_type = self.intermediate_gen.cells[br][bc].collapsed_value
                    if not block_type:
                        continue
                    
                    neighbors = {
                        'top': self._get_block_type(br-1, bc),
                        'bottom': self._get_block_type(br+1, bc),
                        'left': self._get_block_type(br, bc-1),
                        'right': self._get_block_type(br, bc+1)
                    }
                    
                    self.logger.info(f"Generating block ({br},{bc}): {block_type}")
                    local_gen = LocalGenerator(block_type, self.local_size, neighbors)
                    success = local_gen.collapse()
                    
                    if not success:
                        self.logger.error(f"Failed to generate block ({br},{bc})")
                        return False
                    
                    # Save visualization
                    local_map = [[cell.collapsed_value for cell in row] 
                                for row in local_gen.cells]
                    img = self.renderer.render_local_map(local_map, block_type)
                    img.save(local_dir / f"block_{br}_{bc}_{block_type}.png")
                    
                    # Save block data
                    with open(local_dir / f"block_{br}_{bc}.json", 'w') as f:
                        json.dump({
                            "type": block_type,
                            "position": {"row": br, "col": bc},
                            "neighbors": neighbors,
                            "tiles": local_map
                        }, f, indent=2)
                    
                    self._copy_local_to_complete(local_gen, br, bc)
                    
                    # Update progress
                    completed_blocks += 1
                    progress = (completed_blocks / total_blocks) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({completed_blocks}/{total_blocks})")
            
            # Save final complete map with grid overlays
            img = self.renderer.render_final_map(
                self.complete_tilemap,
                self.global_size,
                (self.intermediate_gen.height, self.intermediate_gen.width)
            )
            img.save(self.run_dir / "complete_map.png")
            
            # Save clean version of the complete tilemap without grid overlays
            clean_img = self.renderer.render_complete_tilemap(self.complete_tilemap)
            clean_img.save(self.run_dir / "complete_tilemap_clean.png")
            
            # Save complete tilemap data
            with open(self.run_dir / "complete_tilemap.json", 'w') as f:
                json.dump(self.complete_tilemap, f, indent=2)
            
            return True

        except Exception as e:
            self.logger.exception("Error in local level generation")
            return False

    def _get_block_type(self, row: int, col: int) -> Optional[str]:
        """Get block type at given position, or None if invalid."""
        if (0 <= row < self.intermediate_gen.height and 
            0 <= col < self.intermediate_gen.width):
            return self.intermediate_gen.cells[row][col].collapsed_value
        return None

    def _copy_local_to_complete(self, local_gen: LocalGenerator, 
                              block_row: int, block_col: int) -> None:
        """Copy local tilemap to the complete map."""
        sr = block_row * self.local_size
        sc = block_col * self.local_size
        
        for i in range(self.local_size):
            for j in range(self.local_size):
                value = local_gen.cells[i][j].collapsed_value
                if value:
                    self.complete_tilemap[sr + i][sc + j] = value

def main():
    """Main entry point with improved parameter handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='HAMC Map Generator')
    parser.add_argument('--width', type=int, default=3, help='Global map width')
    parser.add_argument('--height', type=int, default=3, help='Global map height')
    parser.add_argument('--subgrid', type=int, default=2, help='Subgrid size')
    parser.add_argument('--local', type=int, default=4, help='Local block size')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    generator = HAMCMapGenerator(
        global_size=(args.width, args.height),
        subgrid_size=args.subgrid,
        local_size=args.local,
        output_dir=args.output,
        seed=args.seed,
        log_level=logging.DEBUG if args.debug else logging.INFO
    )
    
    if generator.generate():
        print("\nGeneration successful! Check the output directory for results.")
        sys.exit(0)
    else:
        print("\nGeneration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
