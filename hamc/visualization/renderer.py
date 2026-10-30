
from PIL import Image, ImageDraw

from ..config.tile_config import TileConfig


class MapRenderer:
    """Renderer for visualizing different levels of the HAMC map."""

    # Colours kept in sync with the dashboard palette so PNG exports and the
    # live UI render the same scenery.
    REGION_COLORS = {
        "forest": (52, 138, 78),
        "desert": (224, 198, 142),
        "city":   (132, 138, 152),
        "None":   (16, 22, 30),
        "ERROR":  (220, 60, 80),
    }

    BLOCK_COLORS = {
        "grove":       (33, 110, 50),
        "clearing":    (124, 198, 73),
        "river":       (76, 175, 230),
        "sand":        (224, 198, 142),
        "oasis":       (60, 158, 130),
        "residential": (172, 70, 56),
        "commercial":  (170, 210, 230),
        "industrial":  (170, 175, 182),
        "road":        (62, 62, 70),
        "park":        (96, 170, 84),
        "scrubland":   (154, 195, 90),
        "periurban":   (148, 152, 158),
        "village":     (180, 132, 80),
        "None":        (16, 22, 30),
        "ERROR":       (220, 60, 80),
    }

    def __init__(self, tile_size: int = 20, padding: int = 1):
        self.tile_size = tile_size
        self.padding = padding

    def render_global_map(self, global_map: list[list[str]]) -> Image.Image:
        """Render the global level map."""
        return self._render_grid(global_map, self.REGION_COLORS)

    def render_intermediate_map(self, intermediate_map: list[list[str]]) -> Image.Image:
        """Render the intermediate level map."""
        return self._render_grid(intermediate_map, self.BLOCK_COLORS)

    def render_final_map(self, tilemap: list[list[str]],
                        global_size: tuple[int, int],
                        intermediate_size: tuple[int, int]) -> Image.Image:
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

    def render_local_map(self, local_map: list[list[str]], block_type: str) -> Image.Image:
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

    def render_complete_tilemap(self, tilemap: list[list[str]]) -> Image.Image:
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

    def _render_grid(self, grid: list[list[str]],
                    color_map: dict[str, tuple[int, int, int]]) -> Image.Image:
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
