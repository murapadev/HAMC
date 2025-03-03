# Hierarchical Adaptive Model Collapse (HAMC)

## Overview

This repository contains a prototype implementation of procedural content generation that combines:

- **Wave Function Collapse (WFC)** with weighted options to prioritize certain elements
- **Content hierarchies** (global regions, intermediate blocks, and local tiles)
- **Connectivity constraints** specifically for rivers and roads

The approach is inspired by a multi-level generative model where each level (Global, Intermediate, and Local) collapses its cells using a WFC-like method with distinct rules and weights.




## Technical Approach

The system is divided into **three hierarchical layers**:

1. **Global Layer:**  
   Assigns **regions** (e.g., "bosque", "desierto", "ciudad") to cells in a large grid. Defines global-level compatibility rules (which region can be adjacent to which).
   - Uses a dictionary `{region: probability}` to handle **weights** of each region.

2. **Intermediate Layer:**  
   Each global cell is subdivided into a `subgrid` with **blocks** (e.g., "arboleda", "arena", "residencial"). 
   - Includes **transition blocks** (e.g., "matorral", "periurbano") to smooth boundaries between different regions.
   - Defines compatibility rules between blocks and includes **weights** (appearance preferences).

3. **Local Layer (Tilemap):**  
   For each block, generates a small tilemap (e.g., "Hierba", "Agua", "Arena", "Asfalto") using a local WFC with its own rules and **weights**.
   - Incorporates **connectivity constraints**, such as requiring a "rio" block to have a vertical "Agua" path, or a "carretera" block to have a horizontal "Asfalto/Linea" path.

At each level, a "collapse and propagation" algorithm is applied (similar to classic WFC), but integrates an **entropy function** based on the **weights** of each option (using *Shannon entropy*). When a cell collapses, constraints propagate to neighboring cells to discard incompatible options.

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
│   │   ├── cell.py                   # Cell class
│   │   ├── compatibility_cache.py    # Caching for compatibility checks
│   │   ├── entropy.py                # Entropy calculations
│   │   └── generator_state.py        # Generator state management
│   ├── generators/                   # Generator implementations
│   │   ├── base_generator.py         # Abstract base generator
│   │   ├── global_generator.py       # Global level generator
│   │   ├── intermediate_generator.py # Intermediate level generator
│   │   └── local_generator.py        # Local level generator
│   └── visualization/                # Visualization utilities
│       └── renderer.py               # Rendering to images
└── tests/                            # Test suite
    └── test_generators.py            # Generator tests
```

## Key Components

### Configuration with Weights

- Region, block, and tile definitions with weights (probabilities)
- Compatibility rules between elements at each level
- Special rules for transitions between different region types

### Entropy and Weighted Selection

- `shannon_entropy`: Calculates entropy of options with their weights
- `weighted_random_choice`: Selects an option based on probabilistic weights

### Cell Classes

Each level uses a `Cell` class that maintains:
- A dictionary of possible values with weights
- Methods for calculating entropy and collapsing to a single value

### Generation Layers

- **GlobalGenerator**: Manages the grid of global cells (regions)
- **IntermediateGenerator**: Manages the intermediate grid, subdividing each region into blocks
- **LocalGenerator**: Manages the generation of local tilemaps for each block with connectivity constraints

### Special Constraints

- River blocks must maintain a continuous vertical water path
- Road blocks must maintain a continuous horizontal asphalt path
- Transition zones between different region types have special block types

## Usage

1. **Install dependencies**
   ```bash
   pip install pillow
   ```

2. **Run the generator**
   ```bash
   python main.py
   ```

3. **Output**
   The script generates multiple visualization images in the `output` directory:
   - `global_level.png`: The global regions map
   - `intermediate_level.png`: The intermediate blocks map  
   - `complete_map.png`: The final detailed tilemap
   - `local_blocks/`: Individual local block tilemaps

## Customization

You can customize the generation by modifying:

- **Configurations**: Edit files in the `hamc/config/` directory to change probabilities, compatibility rules, and available elements
- **Generator parameters**: Adjust map sizes, subgrid divisions, and other parameters in `main.py`
- **Connectivity constraints**: Modify validation methods in `local_generator.py` to add specific constraints

## Future Enhancements

- More sophisticated global coherence rules
- Advanced backtracking that works across hierarchical levels
- Additional connectivity constraints for complex structures
- Performance optimizations for larger maps

## References

This implementation is inspired by the Wave Function Collapse algorithm and hierarchical procedural generation techniques.