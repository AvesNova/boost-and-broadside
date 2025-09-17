# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Boost & Broadside** is a high-performance 2D naval combat simulation built in Rust with Bevy for reinforcement learning agent training. The project is currently in **very early development** with comprehensive requirements but no implementation yet.

### Key Technologies
- **Core Engine**: Rust + Bevy game engine
- **AI Interface**: Python via PyO3 bindings
- **Physics**: Custom arcade-style system (50Hz fixed timestep)
- **Agent Control**: 10Hz decision rate with action persistence
- **Target Performance**: 1000+ FPS in headless training mode

## Development Commands

### Project Setup (To be implemented in Phase 1)
```bash
# Initialize Rust project with Bevy
cargo init --name boost-and-broadside
cargo add bevy

# Add PyO3 for Python integration (Phase 2)
cargo add pyo3 --features "extension-module"

# Development dependencies
cargo add --dev criterion proptest
```

### Build and Testing (Future commands)
```bash
# Standard build
cargo build

# Release build for performance testing
cargo build --release

# Run tests (comprehensive test suite planned)
cargo test

# Run specific test module
cargo test physics
cargo test collision
cargo test integration

# Performance benchmarks
cargo bench

# Run with headless mode for training
cargo run --release -- --headless

# Run in development mode with rendering
cargo run
```

### Python Integration (Phase 2+)
```bash
# Build Python module
maturin develop

# Run Python integration tests
pytest tests/

# Run example Python agent
python examples/simple_agent.py
```

## Core Architecture

### Dual-Rate System Design
- **Physics Loop**: 50Hz (20ms timesteps) - All physics, collision detection, rendering
- **Agent Decisions**: 10Hz (100ms intervals) - AI agent decision making
- **Action Hold**: Each agent decision held for exactly 5 physics frames

### Ship Control System
Ships accept 6 binary actions per decision cycle:
- **Forward/Backward**: Thrust with energy boost system
- **Left/Right**: Arcade-style turning with persistent turn offset
- **Sharp Turn**: Enhanced maneuverability modifier
- **Shoot**: Projectile firing with cooldown management

### Unique Turn-Offset System
Ships maintain a persistent turn offset angle separating heading direction from velocity direction, enabling:
- Strafing while moving forward
- Reverse flight patterns
- Complex tactical positioning

## Physics System

### Arcade Force Model (50Hz Updates)
```rust
thrust_force = thrust_base × thrust_multiplier × heading_unit_vector
drag_force = -drag_coefficient × velocity × speed
lift_force = lift_coefficient × speed² × velocity_perpendicular × turn_direction
total_acceleration = thrust_force + drag_force + lift_force
```

### Simple Integration (0.02s timestep)
```rust
velocity += acceleration × 0.02
position += velocity × 0.02
```

### Collision Detection
- Circle-circle collision detection (simple distance checks)
- No swept detection needed due to 50Hz frequency
- Collision types: Ship-Projectile, Projectile-Projectile

## AI Agent Interface

### Observations (per ship, 10Hz)
10 values per ship:
- position_x, position_y - Ocean coordinates
- velocity_x, velocity_y - Current velocity components  
- heading_cos, heading_sin - Ship orientation unit vector
- turn_offset - Current turn offset in radians
- energy_normalized - Energy level (0-1)
- health_normalized - Health level (0-1)
- ammo_normalized - Ammunition level (0-1)

### Actions (binary vector, 10Hz)
6 binary actions: `[forward, backward, left, right, sharp_turn, shoot]`

## Development Phases

### Phase 1: Core Simulation + Basic Rendering
- Bevy setup with 50Hz rendering and fixed timestep
- Multi-ship physics system (2-4 ships)
- Arcade maneuvering with turn-offset system
- Projectile system with object pooling
- Circle-circle collision detection
- World boundaries with toroidal wrapping
- Action hold system (5 frames at 50Hz = 10Hz simulation)

### Phase 2: Python Integration  
- PyO3 bindings for `NavalCombatGame` class
- 10Hz Python interface with internal 50Hz processing
- Headless mode for training
- Data marshalling for actions/observations
- Python wrapper and test scripts

### Phase 3: RL Integration
- Gymnasium environment interface
- Multi-agent support (1-8 ships)
- Episode management and statistics
- Stable Baselines3 compatibility

### Phase 4: Configuration & Variety
- Runtime parameter system (JSON/YAML configs)
- Multiple ship types and parameters
- Different world sizes and scenarios
- Obstacle support

### Phase 5: Performance Optimization
- Target 1000+ agent decisions/second (10Hz)
- Memory optimization and object pooling
- Multi-threading where beneficial
- Performance regression testing

### Phase 6: Enhanced Visuals & Polish
- Naval-themed sprites and particle effects
- UI improvements and debug tools
- Visual regression testing

## Key Parameters

### Ship Physics (Default Values)
- `thrust_base`: 10.0
- `forward_boost_multiplier`: 5.0  
- `backward_boost_multiplier`: 0.5
- `max_boost_energy`: 100.0
- `collision_radius`: 10.0

### Maneuvering (Default Values)
- `normal_turn_angle`: 5° (0.087 radians)
- `sharp_turn_angle`: 15° (0.262 radians)
- `normal_turn_drag_coefficient`: 1e-3
- `sharp_turn_drag_coefficient`: 3e-3

### Projectiles (Default Values)
- `max_projectiles`: 16 per ship
- `projectile_speed`: 500.0
- `projectile_lifetime`: 1.0 seconds (50 frames)
- `firing_cooldown`: 0.04 seconds (2 frames)
- `projectile_damage`: 20.0

### World System
- **Default Size**: 1200×800 ocean
- **Topology**: Toroidal (wrap-around)
- **Fleet Support**: 1-8 ships with independent parameters
- **Multi-Agent**: All ships controlled by Python agents

## Testing Strategy

### Unit Testing Focus Areas
- Physics calculations and integration
- Collision detection algorithms
- Energy management (50Hz updates)
- Projectile system mechanics
- Action hold system (10Hz → 5 frames)
- Parameter validation and edge cases
- Dual-rate timing synchronization

### Integration Testing
- Python-Rust interface via PyO3
- Multi-ship interactions and combat
- Episode management and resets
- RL framework compatibility
- Performance benchmarks

### Testing Tools
- `cargo test` for Rust unit tests
- `pytest` for Python integration tests
- `proptest` for property-based testing
- `criterion` for performance benchmarks
- CI/CD with automated regression testing

## Performance Requirements

### Training Mode
- **Target**: 1000+ agent decision cycles/second (10Hz decisions)
- **Physics**: 5000+ physics frames/second (50Hz)
- **Mode**: Headless operation, no rendering overhead

### Human Play Mode  
- **Real-time**: 10 agent decisions/second, 50 physics updates/second
- **Rendering**: Direct 50Hz visual output, no interpolation needed
- **Responsiveness**: Immediate visual feedback

## File Organization (Planned)

```
src/
├── main.rs                 # Entry point and game loop
├── ship.rs                 # Ship entity, physics, and maneuvering
├── projectile.rs           # Projectile system and collision
├── physics.rs              # Core physics calculations  
├── world.rs                # Ocean boundaries and wrapping
├── collision.rs            # Circle-circle collision detection
├── energy.rs               # Energy management system
├── python_interface.rs     # PyO3 bindings and data marshalling
└── config.rs               # Parameter loading and validation

tests/
├── unit/                   # Unit tests for each module
├── integration/            # Python-Rust integration tests  
└── benchmarks/             # Performance benchmarks

python/
├── naval_combat.py         # Python wrapper class
├── examples/               # Example agents and scripts
└── tests/                  # Python integration tests
```

## Development Notes

- Always run the full test suite to ensure changes don't create new bugs
- Focus on arcade-style physics that are fast and fun rather than realistic
- Maintain the dual-rate architecture (50Hz physics, 10Hz agents) throughout development
- Prioritize performance optimization from early phases
- Use comprehensive unit testing for all physics calculations
- Test timing synchronization between 50Hz and 10Hz systems
- Validate all parameter loading and edge cases
- Performance test every major change to maintain 1000+ FPS target