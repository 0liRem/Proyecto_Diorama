# Minecraft Skyblock Ray Tracer

A real-time ray tracer written in Rust that renders a Minecraft skyblock island with photorealistic lighting effects.

## Demo Video

[![Minecraft Skyblock Ray Tracer Demo](https://i9.ytimg.com/vi/Ax7Z29DYfc8/mqdefault.jpg?sqp=CIzM5cYG-oaymwEmCMACELQB8quKqQMa8AEB-AH-BIAC4AOKAgwIABABGCwgRyh_MA8=&rs=AOn4CLAsa0eouozkeyJ-f_stNLAa47qr1g)](https://youtu.be/Ax7Z29DYfc8)


## Features

### World Generation
- Complete floating island with bedrock foundation, dirt layers, and grass surface
- Oak tree with wooden trunk and leaf crown
- Water pool with realistic refraction effects
- Decorative blocks including iron, gold, diamond, glowstone, and ores

### Ray Tracing Effects
- **Reflection**: Metallic blocks (iron, gold) reflect the environment
- **Refraction**: Water, glass, and diamonds bend light realistically
- **Emission**: Glowstone and lava act as light sources
- **Global illumination**: Multiple light sources with soft shadows

### Materials System

The project implements 17 different Minecraft materials:

**Diffuse Materials**: Grass Block, Dirt, Bedrock, Oak Wood, Oak Leaves, Obsidian, Cobblestone, Coal Ore, Iron Ore, Diamond Ore

**Metallic Materials**: Iron Block (reflective), Gold Block (highly reflective)

**Dielectric Materials**: Water (n=1.33), Glass (n=1.5), Diamond Block (n=2.4)

**Emissive Materials**: Glowstone, Lava

## Technical Specifications

### Performance Optimizations
- Adaptive rendering system that adjusts quality based on framerate
- Multi-threaded rendering using Rayon for CPU parallelization
- Dynamic resolution scaling (320x240 base resolution)
- Optimized lighting calculations for real-time performance
- Early exit ray intersection optimizations

### Technologies
- **Language**: Rust
- **Parallelization**: Rayon
- **Window Management**: minifb
- **Image Loading**: image crate
- **Random Generation**: rand crate

## Requirements

- Rust 1.70 or higher
- Modern multi-core CPU recommended
- 4GB RAM minimum
- Windows, Linux, or macOS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/0liRem/Proyecto_Diorama
cd Proyecto_Diorama
```

2. Create textures directory:
```bash
mkdir textures
```

3. Add PNG texture files (optional):
   - grass_block.png, dirt.png, bedrock.png
   - oak_wood.png, oak_leaves.png
   - water.png, glass.png
   - iron_block.png, gold_block.png, diamond_block.png
   - glowstone.png, obsidian.png, cobblestone.png
   - coal_ore.png, iron_ore.png, diamond_ore.png
   - lava.png

4. Build and run:
```bash
cargo run --release
```

## Controls

- **WASD**: Move forward/back/left/right
- **QE**: Move down/up
- **Mouse**: Look around
- **ESC**: Exit

## Performance

The adaptive rendering system automatically adjusts quality:
- FPS < 10: Renders every 4th pixel (maximum speed)
- FPS < 20: Renders every 2nd pixel (balanced)
- FPS ≥ 20: Renders all pixels (maximum quality)

Expected performance: 15-30 FPS on modern hardware at 320x240 resolution.

## Architecture

The codebase is structured with separate modules for:
- Vector mathematics and ray operations
- Camera system with fly-cam controls
- Material definitions and lighting calculations
- Geometric primitives (spheres and cubes)
- World generation and object placement
- Optimized rendering pipeline

## License

This project is licensed under the MIT License.
