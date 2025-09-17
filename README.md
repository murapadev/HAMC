# HAMC · Hierarchical Adaptive Model Collapse


<p align="center">
  <a href="https://github.com/murapadev/hamc"><img alt="Repo" src="https://img.shields.io/badge/repo-murapadev%2Fhamc-24292e?logo=github&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/murapadev/hamc?color=blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-68%20passing-brightgreen?logo=pytest">
  <a href="https://github.com/murapadev/hamc/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/murapadev/hamc?style=social"></a>
</p>

<p align="center">
  Multi‑level procedural generation with WFC, adaptive backtracking, and rich validations.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Approach](#technical-approach)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Running Tests](#running-tests)
- [Customization](#customization)
- [Advanced Configuration 🚀](#advanced-configuration)
- [Future Enhancements](#future-enhancements)
- [References](#references)

## Overview

This repository contains a multi-level procedural content generation system that combines:

- **Wave Function Collapse (WFC)** with weighted options to prioritize certain elements
- **Content hierarchies** (global regions, intermediate blocks, and local tiles)
- **Connectivity constraints** specifically for rivers, roads, and other special structures
- **Adaptive backtracking** for better handling of generation failures

The approach is inspired by a hierarchical generative model where each level (Global, Intermediate, and Local) collapses its cells using a WFC-like method with distinct rules, weights, and constra[...]  

## Features

- **Hierarchical Generation**: Three distinct layers with their own rules and constraints
- **Weighted Options**: Probabilistic selection based on configured weights
- **Shannon Entropy Calculation**: More intelligent cell selection during collapse
- **Path Validation**: Special validation for rivers and roads to ensure connectivity
- **Special Block Types**: Customized generation for specific block types (oasis, buildings, etc.)
- **Transition Zones**: Smooth transitions between different region types
- **Visualization**: Renders the generated world at all levels of detail
- **Adaptive Backtracking**: Intelligent backtracking to recover from generation failures

## Technical Approach

The system is divided into **three hierarchical layers**:

1. **Global Layer:**  
   Assigns **regions** (e.g., "forest", "desert", "city") to cells in a large grid. Defines global-level compatibility rules for region adjacency.

   - Uses a dictionary `{region: probability}` to handle **weights** of each region.

2. **Intermediate Layer:**  
   Each global cell is subdivided into a `subgrid` with **blocks** (e.g., "grove", "sand", "residential").

   - Includes **transition blocks** (e.g., "scrubland", "periurban") to smooth boundaries between different regions.
   - Defines compatibility rules between blocks and includes **weights** (appearance preferences).

3. **Local Layer (Tilemap):**  
   For each block, generates a small tilemap (e.g., "Hierba", "Agua", "Arena", "Asfalto") using a local WFC with its own rules and **weights**.
   - Incorporates **connectivity constraints**, such as requiring a "river" block to have a vertical "Agua" path, or a "road" block to have a horizontal "Asfalto/Linea" path.
   - Special blocks like "oasis" have center-focused water distribution.

At each level, a "collapse and propagation" algorithm is applied with an **entropy function** based on the **weights** of each option (using _Shannon entropy_). When a cell collapses, constraints [...]  

## Project Structure

```
.
├── main.py                           # Main entry point
├── hamc/                             # Core package
│   ├── config/                       # Configuration files
│   │   ├── block_config.py           # Intermediate blocks config
│   │   ├── generator_config.py       # Generator parameters
│   │   ├── region_config.py          # Global regions config
│   │   └── tile_config.py            # Local tiles config
│   ├── core/                         # Core functionality
│   │   ├── backtrack_manager.py      # Advanced backtracking system
│   │   ├── cell.py                   # Cell class
│   │   ├── compatibility_cache.py    # Caching for compatibility checks
│   │   ├── entropy.py                # Entropy calculations
│   │   ├── generator_state.py        # Generator state management
│   │   ├── pattern_detector.py       # Pattern recognition utilities
│   │   └── validator.py              # Path and structure validation
│   ├── generators/                   # Generator implementations
│   │   ├── base_generator.py         # Abstract base generator
│   │   ├── global_generator.py       # Global level generator
│   │   ├── intermediate_generator.py # Intermediate level generator
│   │   ├── local_generator.py        # Local level generator
│   │   └── parallel_generator.py     # Multi-threaded generator
│   └── visualization/                # Visualization utilities
│       └── renderer.py               # Rendering to images
├── tests/                            # Test suite
│   ├── __init__.py
│   └── test_generators.py            # Generator tests
├── run_tests.py                      # Test runner with coverage
└── setup.py                          # Package installation script
```

## Key Components

### Cell Class

Each level uses a `Cell` class that maintains:

- A dictionary of possible values with weights
- Methods for calculating entropy and collapsing to a single value
- Tracking of collapsed state and probability distribution

### Generation Layers

- **BaseGenerator**: Abstract base class with common WFC functionality

  - Common backtracking mechanism
  - Snapshot and restore capabilities
  - Status tracking and logging

- **GlobalGenerator**: Manages the grid of global cells (regions)

  - Region compatibility enforcement
  - Ensures all required region types appear
  - Biome distribution validation

- **IntermediateGenerator**: Manages the intermediate grid of blocks

  - Subdivides each region into blocks
  - Handles transition zones between different region types
  - Special block compatibility rules

- **LocalGenerator**: Manages the generation of local tilemaps
  - Path validation for rivers and roads
  - Special initialization for different block types
  - Connectivity constraints
  - Optimized constraint propagation

### Special Block Handling

- **River blocks**: Maintain continuous vertical water paths with validation
- **Road blocks**: Maintain continuous horizontal asphalt paths
- **Oasis blocks**: Create water concentration in the center with sand transitions
- **Building blocks**: Generate structured residential, commercial, or industrial layouts

### Backtracking System

The `BacktrackManager` provides sophisticated backtracking capabilities:

- Priority-based backtrack point selection
- Adaptive strategies based on success rates
- Failed value tracking to avoid repeating errors
- Level-aware backtracking to optimize performance

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate a quick demo (CLI)**

   ```bash
   # Runs the demo pipeline and saves images to ./output
   python main.py \
     --width 4 \
     --height 3 \
     --subgrid 2 \
     --local 4 \
     --output output \
     --debug
   ```

   Output files (in `output/`):
   - `global_map.png` — global regions
   - `intermediate_map.png` — intermediate blocks
   - `final_tilemap.png` — full tilemap with overlays
   - `final_tilemap_clean.png` — full tilemap without overlays
   - `regions.json`, `blocks.json`, `tilemap.json` — raw data

3. **Generate a quick demo (Python)**

   ```python
   from hamc.generators.global_generator import GlobalGenerator
   from hamc.generators.intermediate_generator import IntermediateGenerator
   from hamc.generators.local_generator import LocalGenerator
   from hamc.visualization.renderer import MapRenderer

   # Params
   Wg, Hg, S, L = 4, 3, 2, 4
   renderer = MapRenderer(tile_size=20, padding=1)

   # Global
   G = GlobalGenerator(Wg, Hg)
   G.initialize(); G.collapse()
   regions = [[c.collapsed_value for c in row] for row in G.cells]
   renderer.render_global_map(regions).save('output/global_map.png')

   # Intermediate
   I = IntermediateGenerator(G, subgrid_size=S)
   I.collapse()
   blocks = [[c.collapsed_value for c in row] for row in I.cells]
   renderer.render_intermediate_map(blocks).save('output/intermediate_map.png')

   # Local stitching (final tilemap)
   Lh, Lw = len(I.cells), len(I.cells[0])
   tilemap = []
   for br in range(Lh):
       row_tiles = []
       for bc in range(Lw):
           t = I.cells[br][bc].collapsed_value
           LG = LocalGenerator(t, L)
           LG.collapse()
           row_tiles.append([[cell.collapsed_value for cell in r] for r in LG.cells])
       # horizontal stitch
       for r in range(L):
           tilemap.append(sum([b[r] for b in row_tiles], []))

   renderer.render_final_map(tilemap, global_size=(Hg, Wg), intermediate_size=(Hg*S, Wg*S))\
           .save('output/final_tilemap.png')
   ```

4. **Output**
   - `output/global_map.png`: global regions
   - `output/intermediate_map.png`: intermediate blocks
    - `output/final_tilemap.png`: full tilemap with grid overlays

## Examples

- CLI via Makefile

  ```bash
  make demo WIDTH=5 HEIGHT=4 SUBGRID=2 LOCAL=4 OUT=output SEED=42
  ```

- CLI directly

  ```bash
  python main.py --width 5 --height 4 --subgrid 2 --local 4 --output output --seed 42
  ```

- Minimal Python (library usage)

  See the “Generate a quick demo (Python)” snippet in Quickstart for an end‑to‑end example.

## Running Tests

Tests can be run with the included test runner:

```bash
python run_tests.py
```

The test suite validates:

- Cell initialization and entropy calculation
- Proper block and region compatibility
- Path validation for special blocks
- Collapse and propagation mechanisms

## Customization

You can customize the generation by modifying:

- **Configurations**: Edit files in the `hamc/config/` directory to change probabilities, compatibility rules, and available elements
- **Generator parameters**: Adjust map sizes, subgrid divisions, and other parameters in `main.py`
- **Connectivity constraints**: Modify validation methods in `validator.py` to add specific constraints

<a id="advanced-configuration"></a>

## Advanced Configuration System 🚀

### Overview

HAMC now features a comprehensive configuration system that eliminates hardcoded values and provides dynamic, profile-based configuration management.

### Key Features

- **Zero Hardcoded Values**: All constants moved to configuration files
- **Profile-Based Configuration**: Different settings for development, production, and testing
- **Dynamic Loading**: Runtime configuration changes without code modification
- **Type Safety**: Full type hints and validation
- **Extensible**: Easy to add new configuration parameters

### Configuration Structure

The configuration is organized into specialized sections:

```python
from hamc.config.advanced_config import get_config, load_config

# Get current configuration
config = get_config()

# Access specific sections
entropy_config = config.entropy
cache_config = config.cache
generator_config = config.generator
validator_config = config.validator
performance_config = config.performance
logging_config = config.logging
```

### Configuration Profiles

#### Development Profile (`development.json`)

- Full debugging enabled
- Moderate performance settings
- Comprehensive logging
- Ideal for development and debugging

#### Production Profile (`production.json`)

- Optimized performance settings
- Minimal logging (WARNING+)
- Large cache sizes
- Parallel processing enabled

#### Testing Profile (`testing.json`)

- Fast execution settings
- Single-threaded processing
- Performance profiling enabled
- Debug logging for test analysis

### Usage Examples

#### Loading Configuration

```python
from hamc.config.advanced_config import load_config, ConfigProfile

# Load from file with specific profile
config = load_config('hamc/config/profiles/production.json', ConfigProfile.PRODUCTION)

# Or use default configuration
config = load_config()
```

#### Accessing Configuration Values

```python
from hamc.config.advanced_config import get_entropy_config, get_cache_config

# Get entropy settings
entropy = get_entropy_config()
temperature = entropy.default_temperature
learning_rate = entropy.default_learning_rate

# Get cache settings
cache = get_cache_config()
max_size = cache.compatibility_cache_max_size
ttl = cache.compatibility_cache_ttl
```

#### Modifying Configuration at Runtime

```python
from hamc.config.advanced_config import get_config

config = get_config()

# Modify entropy settings
config.entropy.default_temperature = 1.2
config.entropy.learning_rate = 0.02

# Modify cache settings
config.cache.compatibility_cache_max_size = 2048
config.cache.compatibility_cache_ttl = 600.0
```

### Configuration Files

Configuration files are stored in `hamc/config/profiles/`:

- `development.json` - Development settings
- `production.json` - Production optimized settings
- `testing.json` - Testing optimized settings

### Migrating from Hardcoded Values

All previously hardcoded values have been moved to configuration:

#### Before (Hardcoded)

```python
class AdaptiveEntropyCalculator:
    def __init__(self, temperature: float = 1.0, learning_rate: float = 0.01):
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.temporal_decay_factor = 0.95  # Hardcoded!
```

#### After (Configuration-based)

```python
from ..config.advanced_config import get_entropy_config

class AdaptiveEntropyCalculator:
    def __init__(self, temperature: Optional[float] = None, learning_rate: Optional[float] = None):
        config = get_entropy_config()
        self.temperature = temperature if temperature is not None else config.default_temperature
        self.learning_rate = learning_rate if learning_rate is not None else config.default_learning_rate
        self.temporal_decay_factor = config.temporal_decay_factor  # From config!
```

### Performance Impact

The configuration system has minimal performance impact:

- **Memory**: ~2KB additional memory usage
- **CPU**: Negligible overhead (< 0.1ms per access)
- **Startup**: Fast loading from JSON files
- **Runtime**: Cached configuration objects

### Best Practices

1. **Use Configuration for Constants**: Any magic numbers should be in configuration
2. **Profile-Specific Settings**: Use appropriate profiles for different environments
3. **Type Hints**: Always use proper type hints for configuration access
4. **Validation**: Validate configuration values on load
5. **Documentation**: Document all configuration parameters

### Extending Configuration

To add new configuration parameters:

1. Add to the appropriate dataclass in `advanced_config.py`
2. Update the JSON profile files
3. Use the configuration in your code
4. Update documentation

Example:

```python
@dataclass
class EntropyConfig:
    # Existing parameters...
    new_parameter: float = 0.5  # Add new parameter with default
```

This configuration system provides the foundation for the revolutionary improvements planned in the HAMC roadmap, enabling dynamic behavior modification without code changes.

## Future Enhancements

- Better performance for large-scale maps
- Additional special block types and constraints
- User interface for real-time generation visualization
- Integration with game engines
- Cross-level constraint propagation

## References

This implementation is inspired by the Wave Function Collapse algorithm and hierarchical procedural generation techniques used in modern game development.
