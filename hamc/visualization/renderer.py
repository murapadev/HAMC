from typing import List, Tuple, Dict
from PIL import Image, ImageDraw
from ..config.tile_config import TileConfig

class MapRenderer:
    """Renderer for visualizing different levels of the HAMC map."""
    
    # Color mapping for global regions
    REGION_COLORS = {
        "bosque":   (34, 139, 34),    # Forest Green
        "desierto": (238, 214, 175),  # Sand
        "ciudad":   (128, 128, 128),  # Gray
        "None":     (0, 0, 0),        # Black
        "ERROR":    (255, 0, 0)       # Red
    }

    # Color mapping for intermediate blocks (using representative colors)
    BLOCK_COLORS = {
        "arboleda":     (0, 100, 0),      # Dark Green
        "claro":        (124, 252, 0),     # Lawn Green
        "rio":          (0, 191, 255),     # Deep Sky Blue
        "arena":        (238, 214, 175),   # Sand
        "oasis":        (32, 178, 170),    # Light Sea Green
        "residencial":  (139, 69, 19),     # Saddle Brown
        "comercial":    (176, 224, 230),   # Powder Blue
        "industrial":   (192, 192, 192),   # Silver
        "carretera":    (70, 70, 70),      # Dark Gray
        "matorral":     (154, 205, 50),    # Yellow Green
        "periurbano":   (169, 169, 169),   # Dark Gray
        "aldea":        (205, 133, 63),    # Peru
        "None":         (0, 0, 0),         # Black
        "ERROR":        (255, 0, 0)        # Red
    }

    def __init__(self, tile_size: int = 20, padding: int = 1):
        self.tile_size = tile_size
        self.padding = padding

    def render_global_map(self, global_map: List[List[str]]) -> Image.Image:
        """Render the global level map."""
        return self._render_grid(global_map, self.REGION_COLORS)

    def render_intermediate_map(self, intermediate_map: List[List[str]]) -> Image.Image:
        """Render the intermediate level map."""
        return self._render_grid(intermediate_map, self.BLOCK_COLORS)

    def render_final_map(self, tilemap: List[List[str]], 
                        global_size: Tuple[int, int],
                        intermediate_size: Tuple[int, int]) -> Image.Image:
        """Render the final tiled map with grid overlays."""
        height, width = len(tilemap), len(tilemap[0])
        img_height = height * self.tile_size
        img_width = width * self.tile_size
        
        # Create base image
        img = Image.new('RGB', (img_width, img_height), color='black')
        draw = ImageDraw.Draw(img)
        
        # Draw tiles
        for row in range(height):
            for col in range(width):
                tile = tilemap[row][col]
                color = TileConfig.get_color(tile)
                x1 = col * self.tile_size
                y1 = row * self.tile_size
                x2 = x1 + self.tile_size - self.padding
                y2 = y1 + self.tile_size - self.padding
                draw.rectangle([x1, y1, x2, y2], fill=color)

        # Draw intermediate grid
        int_height, int_width = intermediate_size
        cell_height = height // int_height
        cell_width = width // int_width
        
        for i in range(int_height + 1):
            y = i * cell_height * self.tile_size
            draw.line([(0, y), (img_width, y)], fill=(200, 200, 200), width=2)
        
        for j in range(int_width + 1):
            x = j * cell_width * self.tile_size
            draw.line([(x, 0), (x, img_height)], fill=(200, 200, 200), width=2)

        # Draw global grid
        glob_height, glob_width = global_size
        region_height = height // glob_height
        region_width = width // glob_width
        
        for i in range(glob_height + 1):
            y = i * region_height * self.tile_size
            draw.line([(0, y), (img_width, y)], fill=(255, 255, 255), width=3)
        
        for j in range(glob_width + 1):
            x = j * region_width * self.tile_size
            draw.line([(x, 0), (x, img_height)], fill=(255, 255, 255), width=3)

        return img

    def render_local_map(self, local_map: List[List[str]], block_type: str) -> Image.Image:
        """Render a single local level block."""
        height, width = len(local_map), len(local_map[0])
        img_height = height * self.tile_size
        img_width = width * self.tile_size
        
        # Create base image
        img = Image.new('RGB', (img_width, img_height), color='black')
        draw = ImageDraw.Draw(img)
        
        # Draw tiles
        for row in range(height):
            for col in range(width):
                tile = local_map[row][col]
                color = TileConfig.get_color(tile)
                x1 = col * self.tile_size
                y1 = row * self.tile_size
                x2 = x1 + self.tile_size - self.padding
                y2 = y1 + self.tile_size - self.padding
                draw.rectangle([x1, y1, x2, y2], fill=color)
                
        return img

    def render_complete_tilemap(self, tilemap: List[List[str]]) -> Image.Image:
        """Render the complete tilemap without grid overlays.
        
        Args:
            tilemap: 2D list of tile types
            
        Returns:
            Image.Image: Image of the complete tilemap
        """
        height, width = len(tilemap), len(tilemap[0])
        img_height = height * self.tile_size
        img_width = width * self.tile_size
        
        # Create base image
        img = Image.new('RGB', (img_width, img_height), color='black')
        draw = ImageDraw.Draw(img)
        
        # Draw tiles
        for row in range(height):
            for col in range(width):
                tile = tilemap[row][col]
                color = TileConfig.get_color(tile)
                x1 = col * self.tile_size
                y1 = row * self.tile_size
                x2 = x1 + self.tile_size - self.padding
                y2 = y1 + self.tile_size - self.padding
                draw.rectangle([x1, y1, x2, y2], fill=color)
                
        return img

    def _render_grid(self, grid: List[List[str]], 
                    color_map: Dict[str, Tuple[int, int, int]]) -> Image.Image:
        """Helper method to render a grid with given color mapping."""
        height, width = len(grid), len(grid[0])
        img_height = height * self.tile_size
        img_width = width * self.tile_size
        
        # Create image
        img = Image.new('RGB', (img_width, img_height), color='black')
        draw = ImageDraw.Draw(img)
        
        # Draw cells
        for row in range(height):
            for col in range(width):
                value = grid[row][col] or "None"
                color = color_map.get(value, color_map["ERROR"])
                x1 = col * self.tile_size
                y1 = row * self.tile_size
                x2 = x1 + self.tile_size - self.padding
                y2 = y1 + self.tile_size - self.padding
                draw.rectangle([x1, y1, x2, y2], fill=color)
        
        return img
