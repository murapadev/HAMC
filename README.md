# Hierarchical Adaptive Model Collapse (HAMC)

## Overview

This repository contains a multi-level procedural content generation system that combines:

- **Wave Function Collapse (WFC)** with weighted options to prioritize certain elements
- **Content hierarchies** (global regions, intermediate blocks, and local tiles)
- **Connectivity constraints** specifically for rivers, roads, and other special structures
- **Adaptive backtracking** for better handling of generation failures

The approach is inspired by a hierarchical generative model where each level (Global, Intermediate, and Local) collapses its cells using a WFC-like method with distinct rules, weights, and constraints.

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
   Assigns **regions** (e.g., "bosque", "desierto", "ciudad") to cells in a large grid. Defines global-level compatibility rules for region adjacency.
   - Uses a dictionary `{region: probability}` to handle **weights** of each region.

2. **Intermediate Layer:**  
   Each global cell is subdivided into a `subgrid` with **blocks** (e.g., "arboleda", "arena", "residencial"). 
   - Includes **transition blocks** (e.g., "matorral", "periurbano") to smooth boundaries between different regions.
   - Defines compatibility rules between blocks and includes **weights** (appearance preferences).

3. **Local Layer (Tilemap):**  
   For each block, generates a small tilemap (e.g., "Hierba", "Agua", "Arena", "Asfalto") using a local WFC with its own rules and **weights**.
   - Incorporates **connectivity constraints**, such as requiring a "rio" block to have a vertical "Agua" path, or a "carretera" block to have a horizontal "Asfalto/Linea" path.
   - Special blocks like "oasis" have center-focused water distribution.

At each level, a "collapse and propagation" algorithm is applied with an **entropy function** based on the **weights** of each option (using *Shannon entropy*). When a cell collapses, constraints propagate to neighboring cells, eliminating incompatible options.

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

## Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the generator**
   ```bash
   python main.py [--width WIDTH] [--height HEIGHT] [--subgrid SUBGRID] [--local LOCAL] [--seed SEED] [--output OUTPUT] [--debug]
   ```

   Parameters:
   - `--width`: Global map width (default: 3)
   - `--height`: Global map height (default: 3)
   - `--subgrid`: Intermediate subgrid size (default: 2)
   - `--local`: Local block size (default: 4)
   - `--seed`: Random seed for reproducible generation
   - `--output`: Output directory (default: 'output')
   - `--debug`: Enable debug logging

3. **Output**
   The script generates multiple visualization images in the specified output directory:
   - `global_level.png`: The global regions map
   - `intermediate_level.png`: The intermediate blocks map  
   - `complete_map.png`: The final detailed tilemap with grid overlays
   - `complete_tilemap_clean.png`: The final tilemap without grid overlays
   - `local_blocks/`: Individual local block tilemaps
   - Various JSON files with the raw map data

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

## Future Enhancements

- Better performance for large-scale maps
- Additional special block types and constraints
- User interface for real-time generation visualization
- Integration with game engines
- Cross-level constraint propagation

## References

This implementation is inspired by the Wave Function Collapse algorithm and hierarchical procedural generation techniques used in modern game development.