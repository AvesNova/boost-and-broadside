# Boost & Broadside

A high-performance 2D naval combat simulation built in Rust with Bevy for reinforcement learning agent training.

## 🚀 Project Status

**🚧 In Development** - This project is currently in very early development.

## 🎯 Overview

Boost & Broadside is a real-time physics-based naval combat simulation designed specifically for training reinforcement learning agents. Ships engage in tactical combat using arcade-style physics that prioritize fun, interesting gameplay mechanics over realism.

### Key Features (Planned)

- **High-Performance Simulation**: Target 1000+ FPS in headless training mode
- **Arcade Physics**: Fast, responsive ship maneuvering with unique turn-offset mechanics
- **Python AI Control**: Seamless PyO3 integration for neural network and script-based agents
- **Energy Management**: Strategic resource allocation for boost capabilities
- **Real-Time Combat**: Projectile-based combat with cooldowns and ammunition systems
- **Configurable Parameters**: Extensive customization for different training scenarios

## 🏗️ Architecture

```
┌─────────────────┐    PyO3   ┌──────────────────────┐
│   Python AI     │◄─────────►│    Rust Engine       │
│   - Neural Nets │           │   - Bevy ECS         │
│   - Scripts     │           │   - Physics (50Hz)   │
│   - RL Training │           │   - Rendering (50fps)│
└─────────────────┘           └──────────────────────┘
```

### Technology Stack

- **Core Engine**: Rust + Bevy game engine
- **AI Interface**: Python via PyO3 bindings
- **Physics**: Custom arcade-style system (50Hz fixed timestep)
- **Agent Control**: 10Hz decision rate with action persistence
- **Rendering**: 50FPS visuals (optional in training)

## 🎮 Gameplay Mechanics

### Ship Control
Ships are controlled through 6 binary actions:
- **Forward/Backward**: Thrust with energy boost system
- **Left/Right**: Arcade-style turning with persistent turn offset
- **Sharp Turn**: Enhanced maneuverability modifier
- **Shoot**: Projectile firing with cooldown management

### Unique Turn-Offset System
Ships maintain a turn offset angle that separates heading direction from velocity direction, enabling advanced maneuvers like:
- Strafing while moving forward
- Reverse flight patterns
- Complex tactical positioning

### Energy Management
- Boost capabilities consume energy resources
- Strategic energy allocation affects combat effectiveness
- Energy regeneration creates tactical decision points

## 🤖 AI Agent Interface

### Observations (per ship, 10Hz)
```python
{
    'position': [x, y],
    'velocity': [vx, vy], 
    'heading': [cos_θ, sin_θ],
    'turn_offset': float,
    'energy_normalized': float,
    'health_normalized': float,
    'ammo_normalized': float
}
```

### Actions (binary vector, 10Hz)
```python
[forward, backward, left, right, sharp_turn, shoot]
```

## 🎯 Use Cases

- **Reinforcement Learning Research**: Multi-agent combat scenarios
- **AI Training**: Naval tactics and maneuvering strategies  
- **Simulation Testing**: Physics and control system validation
- **Educational**: Game development and AI integration learning

## 📋 Development Roadmap

### Phase 1: Core Engine
- [ ] Bevy project setup and basic ECS architecture
- [ ] Physics system implementation (50Hz fixed timestep)
- [ ] Ship entity system with arcade maneuvering
- [ ] Basic rendering and visualization

### Phase 2: Combat System
- [ ] Projectile system with collision detection
- [ ] Energy and resource management
- [ ] Health and damage systems
- [ ] Combat balancing and testing

### Phase 3: Python Integration
- [ ] PyO3 bindings for ship control
- [ ] Observation and action interfaces
- [ ] Python API design and testing
- [ ] Performance optimization

### Phase 4: AI Integration
- [ ] Gymnasium environment wrapper
- [ ] Multi-agent training scenarios
- [ ] Example RL training scripts
- [ ] Documentation and tutorials

### Phase 5: Polish & Performance
- [ ] Headless training mode optimization
- [ ] Configuration system for scenarios
- [ ] Comprehensive testing suite
- [ ] Performance benchmarking tools

## 🚀 Getting Started

*Installation and usage instructions will be added as development progresses.*

## 🤝 Contributing

This project is developed by Avesnova. Contribution guidelines will be established once the core architecture is implemented.

## 📄 License

*License information will be added.*

## 🔗 Related Projects

- [Bevy Game Engine](https://bevyengine.org/)
- [PyO3 - Rust Python Bindings](https://pyo3.rs/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

---

**Built by [Avesnova](https://github.com/avesnova)** 🌊⚓
