# Naval Combat Game - Requirements and Development Plan

## Project Overview

### Purpose
This document outlines the requirements and development plan for a real-time physics-based 2D naval combat simulation designed for reinforcement learning agent training. The system will be implemented using Rust with the Bevy game engine, featuring a Python control interface via PyO3 for AI agent control.

### Theme and Setting
The game features naval vessels (ships) engaged in combat on a 2D ocean surface. Ships maneuver using arcade-style physics optimized for fun and interesting gameplay rather than realism. The physics are designed to be fast to compute while providing engaging tactical combat mechanics. All agents are controlled through Python neural networks or scripts, with emphasis on runtime performance and development efficiency.

### Scope
The project encompasses a complete simulation environment where naval vessels engage in combat using fast, arcade-style physics. Ships can fire projectiles, take damage, and must manage energy resources. All ship agents will be controlled via Python, with the Rust engine handling physics, rendering, and core gameplay logic.

## System Architecture

### Core Technology Stack
- **Primary Language**: Rust for performance-critical simulation and rendering
- **Game Engine**: Bevy for entity management and rendering
- **Python Integration**: PyO3 for seamless Python-Rust control interface
- **AI Control**: All ship agents controlled via Python scripts or neural networks
- **Testing**: Comprehensive unit and integration test coverage

### Design Philosophy
- **Rust**: Complete game implementation (physics, rendering, gameplay logic)
- **Python**: Pure control layer - sends commands, receives observations
- **Unified Architecture**: Same code path for training and human play modes
- **Arcade Physics**: Fast, fun, and interesting rather than realistic
- **Test-Driven Development**: Extensive unit and integration testing throughout

### Performance Priorities
1. **Runtime Speed**: Optimized for maximum simulation throughput (1000+ FPS target)
2. **Development Speed**: Rapid iteration and testing capabilities
3. **Clean Interface**: Minimal Python-Rust boundary crossings
4. **Fast Physics**: Arcade-style calculations optimized for speed and fun

## Core Game Loop and Timing

### Dual-Rate System Architecture
- **Physics/Game Loop**: 50 Hz (20ms timesteps) - All physics, collision detection, and rendering
- **Agent Decision Rate**: 10 Hz (100ms intervals) - AI agent decision making
- **Action Hold Duration**: 5 frames - Each agent decision is held for 5 physics frames

### Timing Benefits
- **Simple Collision Detection**: 50 Hz physics eliminates need for swept collision detection
- **Stable Physics**: Higher frequency provides better numerical stability
- **Smooth Rendering**: Direct 50 Hz rendering with no interpolation needed
- **Agent Consistency**: 10 Hz decisions maintain consistent AI behavior across modes

### Game Loop Sequence
**Every 20ms (50 Hz):**
1. **Action Processing**: Use current held actions (refreshed every 5 frames at 10 Hz)
2. **Physics Update**: Calculate ship forces, integration, and attitude changes
3. **Projectile Management**: Update positions, handle lifetime expiration
4. **Collision Processing**: Simple circle-circle collision detection
5. **Resource Management**: Update energy, ammunition, and health systems
6. **Rendering**: Update visuals (if not headless)

**Every 100ms (10 Hz):**
7. **Observation Generation**: Prepare state data for AI agents
8. **Action Update**: Receive new actions from Python agents
9. **Episode Management**: Check termination conditions

## Ship Control System

### Action Input Specification
Each ship accepts 6 binary actions per decision cycle (held for 5 frames):
- **Forward**: Apply forward boost multiplier to thrust
- **Backward**: Apply backward boost multiplier to thrust
- **Left**: Engage left turn maneuver
- **Right**: Engage right turn maneuver
- **Sharp Turn**: Modifier for enhanced turning capability
- **Shoot**: Attempt projectile firing

### Action Hold System
- **Decision Frequency**: Agents make decisions at 10 Hz
- **Hold Duration**: Each action set is held for exactly 5 physics frames (100ms)
- **Action Consistency**: Same actions applied across all 5 frames
- **Frame Alignment**: Agent decisions synchronized with physics frames

### Arcade Maneuvering System
Ships maintain a persistent turn offset angle determining the deviation between ship heading and velocity direction. This system is designed for fun, responsive gameplay rather than realistic naval physics.

**Turn Offset Assignment Rules**:
- No inputs: Face velocity direction (0° offset)
- Left only: Apply normal left turn angle (-normal_turn_angle)
- Right only: Apply normal right turn angle (+normal_turn_angle)
- Both left and right: Maintain current turn offset
- Sharp modifier: Use enhanced sharp_turn_angle parameters

**Attitude Calculation**:
```
ship_heading = velocity_direction + turn_offset
```

This system enables advanced arcade maneuvers including strafing, reverse flight, and tactical positioning.

## Physics System (Arcade-Style)

### High-Frequency Physics Benefits
- **50 Hz Update Rate**: Provides stable, smooth physics simulation
- **Simple Integration**: No need for complex collision prediction
- **Circle-Circle Collision**: Fast distance-based detection sufficient at 50 Hz
- **Stable Forces**: Higher frequency prevents physics instabilities

### Arcade Force Model
Designed for fast computation and engaging gameplay:
```
thrust_force = thrust_base × thrust_multiplier × heading_unit_vector
drag_force = -drag_coefficient × velocity × speed
lift_force = lift_coefficient × speed² × velocity_perpendicular × turn_direction
total_acceleration = thrust_force + drag_force + lift_force
```

### Simple Integration (50 Hz timestep = 0.02s)
```
velocity += acceleration × 0.02
position += velocity × 0.02
```

### Energy Management System (Updated every frame)
```
energy_cost = lookup_table[forward_action][backward_action]
energy = clamp(energy - energy_cost × 0.02, 0, max_energy)
thrust_multiplier = (energy > 0) ? boost_multiplier : 1.0
```

## Ship Parameters

### Physics Parameters (Default Values)
- **thrust_base**: 10.0 - Continuous thrust force
- **forward_boost_multiplier**: 5.0 - Forward thrust enhancement
- **backward_boost_multiplier**: 0.5 - Reverse thrust capability
- **max_boost_energy**: 100.0 - Maximum energy capacity
- **base_energy_cost**: -10.0 - Energy regeneration rate
- **forward_energy_cost**: 50.0 - Forward thrust energy consumption
- **backward_energy_cost**: -20.0 - Reverse thrust energy effect

### Arcade Maneuvering Parameters (Default Values)
- **no_turn_drag_coefficient**: 8e-4 - Straight movement drag
- **normal_turn_drag_coefficient**: 1e-3 - Normal turn additional drag
- **sharp_turn_drag_coefficient**: 3e-3 - Sharp turn additional drag
- **normal_turn_lift_coefficient**: 15e-3 - Normal turn lift generation
- **sharp_turn_lift_coefficient**: 30e-3 - Sharp turn lift generation
- **normal_turn_angle**: 5° (in radians) - Standard turn offset
- **sharp_turn_angle**: 15° (in radians) - Enhanced turn offset

### Physical Properties (Default Values)
- **collision_radius**: 10.0 - Collision detection boundary
- **max_health**: 100.0 - Maximum health points

## Projectile System

### Projectile Parameters (Default Values)
- **max_projectiles**: 16 - Maximum projectiles per ship
- **projectile_speed**: 500.0 - Projectile velocity magnitude
- **projectile_damage**: 20.0 - Damage per successful hit
- **projectile_lifetime**: 1.0 seconds (50 frames at 50 Hz)
- **firing_cooldown**: 0.04 seconds (2 frames at 50 Hz)
- **max_ammo**: 32.0 - Maximum ammunition capacity
- **ammo_regen_rate**: 4.0 - Ammunition restoration per second
- **projectile_spread_angle**: 3° (in radians) - Firing accuracy variation

### Projectile Behavior
- **Constant Velocity**: Projectiles maintain initial velocity
- **Frame-Based Lifetime**: Tracked in physics frames for precision
- **Impact Destruction**: Destroyed upon collision with ships or other projectiles
- **Spread Pattern**: Configurable firing accuracy variation

## Collision Detection (Simplified)

### Circle-Circle Collision Detection
- **Simple Distance Checks**: Fast collision detection at 50 Hz eliminates tunneling
- **No Swept Detection Needed**: High frequency physics prevents fast-moving objects from missing collisions
- **Frame-Accurate**: Collisions detected and resolved within single physics frame

### Collision Types
- **Ship-Projectile**: Apply damage to ship, destroy projectile
- **Projectile-Projectile**: Destroy both projectiles
- **Future Enhancement**: Ship-obstacle collision support

## World System (Ocean Environment)

### Ocean Boundaries
- **Toroidal Topology**: Wrap-around behavior at world edges
- **Configurable Dimensions**: Default 1200×800, runtime adjustable
- **Velocity Preservation**: Ships maintain momentum when wrapping

### Fleet Support
- **Ship Count**: Support 1-8 ships with independent parameter sets
- **Multi-Ship from Start**: Multiple ships present from Phase 1
- **Asymmetric Scenarios**: Configurable ship parameters for varied engagements
- **Python Control**: All ships controlled by Python agents

## AI Agent Interface

### Observation Format
Per ship observation vector (10 values, generated at 10 Hz):
- position_x, position_y - Ocean coordinates
- velocity_x, velocity_y - Current velocity components
- heading_cos, heading_sin - Ship orientation unit vector
- turn_offset - Current turn offset in radians
- energy_normalized - Energy level (0-1 scale)
- health_normalized - Health level (0-1 scale)
- ammo_normalized - Ammunition level (0-1 scale)

### Action Processing
- **Binary Action Vectors**: Processed at 10 Hz decision rate
- **Action Hold System**: Each decision held for exactly 5 physics frames
- **Fleet Control**: Support for controlling partial ship populations
- **Python Integration**: All ships controlled via Python agents or scripts

## Performance Requirements

### Training Mode
- **High-Speed Processing**: Target 1000+ decision cycles per second (10 Hz agent decisions)
- **Physics Throughput**: Target 5000+ physics frames per second (50 Hz physics)
- **Headless Operation**: No rendering overhead during training
- **Hardware Optimization**: Maximum simulation throughput

### Human Play Mode
- **Real-Time Synchronization**: 10 agent decisions per second, 50 physics updates per second
- **Direct Rendering**: 50 Hz visual output, no interpolation needed
- **Responsive Controls**: Immediate visual feedback from physics updates

## Testing Strategy

### Unit Testing
- **Physics Calculations**: Test all force calculations, integration, and edge cases
- **Collision Detection**: Verify simple circle-circle collision algorithms
- **Energy Management**: Test energy consumption and regeneration logic (50 Hz updates)
- **Projectile System**: Test firing, movement, and destruction mechanics (50 Hz updates)
- **Action Hold System**: Test 10 Hz agent decisions held for 5 frames
- **Parameter Validation**: Test parameter loading and validation
- **Boundary Conditions**: Test world wrapping and edge cases
- **Timing Systems**: Test 50 Hz/10 Hz synchronization

### Integration Testing
- **Python-Rust Interface**: Test PyO3 bindings and data marshalling
- **Multi-Ship Interactions**: Test ship-to-ship collision and combat
- **Episode Management**: Test reset, termination, and statistics
- **Performance Benchmarks**: Automated performance regression testing
- **RL Framework Integration**: Test Gymnasium and Stable Baselines3 compatibility
- **Timing Integration**: Test dual-rate system behavior

### Testing Tools and Practices
- **Cargo Test**: Standard Rust unit testing framework
- **Pytest**: Python integration testing
- **Property-Based Testing**: Use proptest for edge case discovery
- **Benchmark Testing**: Criterion for performance measurements
- **CI/CD**: Automated testing on every commit
- **Coverage Reporting**: Track test coverage metrics

## Development Plan

### Phase 1: Core Simulation + Basic Rendering
**Goal:** Working multi-ship physics with visual feedback at dual rates

1. **Basic Bevy setup & 50 Hz rendering**
   - 2D camera, window, basic shapes for ships/projectiles
   - 50 Hz game loop with fixed timestep

2. **Core data structures**
   - `Ship`, `Projectile`, `GameState` structs with multiple ships support
   - Action hold system for 5-frame duration

3. **Multi-ship initialization**
   - Spawn 2-4 ships at different positions
   - Only ship 0 is controllable, others are AFK (no actions)

4. **Ship physics implementation (50 Hz)**
   - Force calculations, integration, turn control
   - Apply physics to all ships every 20ms

5. **Basic input system (10 Hz simulation)**
   - Keyboard controls for ship 0 only
   - Actions held for 5 frames to simulate 10 Hz agent behavior

6. **Projectile system (50 Hz)**
   - Firing, movement, lifetime tracking (frame-based)
   - Object pooling for all ships

7. **Simple collision detection (50 Hz)**
   - Circle-circle checks, damage application
   - No swept detection needed

8. **World boundaries**
   - Toroidal wrapping for all entities

9. **Unit tests**
   - Test physics calculations, collision detection, dual-rate timing

**Deliverable:** Playable multi-ship game at 50 Hz physics with 10 Hz control simulation, comprehensive unit tests

### Phase 2: Python Integration
**Goal:** Python control with optional headless mode and dual-rate system

10. **PyO3 bindings**
    - Export `NavalCombatGame` class
    - `step()` method operating at 10 Hz agent rate
    - Internal 50 Hz physics handling

11. **Data marshalling**
    - Actions in (Vec<Vec<bool>>), observations out (Vec<Vec<f32>>)
    - 10 Hz Python interface with 50 Hz internal processing

12. **Headless mode**
    - Optional rendering disable for training
    - Maintain 50 Hz physics, 10 Hz agent interface

13. **Python wrapper**
    - Basic Python class interface
    - Test scripts to control ships at 10 Hz

14. **Integration tests**
    - Test Python-Rust interface, data conversion, headless mode
    - Test dual-rate system behavior

**Deliverable:** Python-controlled simulation with headless capability, dual-rate architecture, full integration test suite

### Phase 3: RL Integration
**Goal:** Complete RL training environment with optimal timing

15. **Gymnasium interface**
    - Standard RL environment API at 10 Hz
    - Internal 50 Hz physics abstracted

16. **Multi-agent support**
    - Control all 1-8 ships independently at 10 Hz

17. **Episode management**
    - Termination conditions, statistics, proper resets

18. **RL framework compatibility**
    - Stable Baselines3 integration, vectorized environments

19. **RL integration tests**
    - Test Gymnasium interface, multi-agent scenarios, episode management

**Deliverable:** Ready-to-use RL training environment with optimal timing architecture

### Phase 4: Configuration & Variety
**Goal:** Support for various scenarios and content

20. **Parameter system**
    - Runtime configuration loading, JSON/YAML configs

21. **Multiple ship types**
    - Different ship parameters and capabilities

22. **Map/world variations**
    - Different world sizes, obstacle support, game modes

23. **Scenario creation**
    - Easy setup of different training scenarios

24. **Configuration tests**
    - Test parameter loading, validation, scenario creation

**Deliverable:** Configurable scenarios for diverse training with test coverage

### Phase 5: Performance Optimization
**Goal:** Hit the 1000+ FPS training target (10 Hz agent decisions)

25. **Performance profiling & optimization**
    - Identify bottlenecks in dual-rate system
    - Optimize 50 Hz physics loop

26. **Memory optimization**
    - Efficient allocation patterns, object pooling

27. **Parallel processing**
    - Multi-threading where beneficial

28. **Performance benchmarks**
    - Automated performance regression testing

**Deliverable:** High-performance simulation meeting speed requirements with performance test suite

### Phase 6: Enhanced Visuals & Polish
**Goal:** Production-quality presentation

29. **Advanced rendering**
    - Naval-themed sprites/graphics, water effects, particle systems

30. **UI improvements**
    - HUD, menus, parameter tweaking interface

31. **Visual debugging tools**
    - Debug overlays, performance metrics display

32. **Polish testing**
    - User experience testing, visual regression testing

**Deliverable:** Polished, visually appealing naval combat game with complete test coverage

## External Interface Requirements

### Python Integration Specifications
- **Data Format**: Flat arrays for actions (boolean) and observations (float32)
- **Episode Management**: Complete episode lifecycle control and statistics
- **PyO3 Implementation**: Seamless Rust-Python interoperability
- **Timing Interface**: 10 Hz Python interface with 50 Hz internal processing

### Configuration Management
- **Runtime Configuration**: All physics and game parameters adjustable
- **World Flexibility**: Support for different ocean sizes and ship counts
- **Episode Control**: Configurable episode length limits and termination conditions
- **Parameter Loading**: Ability to instantiate ships with diverse parameter sets

## Success Criteria

The project will be considered successful when:
1. Simulation achieves target performance of 1000+ agent decision cycles per second (10 Hz)
2. Physics runs smoothly at 50 Hz with simple collision detection
3. Python agents can successfully control naval vessels with responsive behavior
4. Physics simulation provides fast, fun, and engaging arcade-style combat dynamics
5. Configuration system allows rapid experimentation with different naval scenarios
6. Integration with standard RL frameworks (PyTorch, Stable Baselines3) functions seamlessly
7. Comprehensive test suite provides confidence in system reliability and performance

## Deliverables

### Technical Deliverables
- Complete Rust-based naval simulation engine using Bevy with dual-rate architecture
- PyO3 integration layer for Python control
- Comprehensive parameter configuration system
- Performance monitoring and benchmarking tools
- Documentation and API reference

### Testing and Training Deliverables
- Comprehensive unit test suite for all core simulation components
- Integration test suite for Python-Rust interface
- Performance benchmarks and automated regression testing
- Example Python agents and naval training scripts
- Sample naval combat scenarios for RL training
- Test coverage reports and continuous integration setup