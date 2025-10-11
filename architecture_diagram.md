# Hydra Integration Architecture Diagram

## Current Architecture vs. New Architecture

### Current Architecture
```
┌─────────────────┐
│   main.py       │
│  (argparse)     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Pipeline Modules│
│  (load_config)  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│unified_training │
│     .yaml       │
└─────────────────┘
```

### New Hydra-Based Architecture
```
┌─────────────────┐
│   main.py       │
│ (Hydra +        │
│  argparse)      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Hydra Core    │
│ (Compose &      │
│  Merge Configs) │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Pipeline Modules│
│  (DictConfig)   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Config Files    │
│ (base.yaml +    │
│  overrides)     │
└─────────────────┘
```

## Configuration Flow

### Command Execution Flow
```
User Command: python main.py train bc model.transformer.embed_dim=128
         │
         ▼
┌─────────────────┐
│   main.py       │
│ - Parse command │
│ - Detect "train bc" │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Hydra Core    │
│ - Load base.yaml│
│ - Apply train/bc.yaml │
│ - Apply CLI overrides │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  DictConfig     │
│ (Merged config) │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ TrainingPipeline│
│ _train_bc()     │
│ (Uses cfg.model │
│  .transformer)  │
└─────────────────┘
```

## Configuration File Structure

```
src/config/
├── base.yaml              # All base configuration
├── train/                 # Training command overrides
│   ├── bc.yaml           # BC training specific
│   ├── rl.yaml           # RL training specific
│   └── full.yaml         # Full pipeline specific
├── collect/               # Data collection overrides
│   ├── bc.yaml           # BC data collection
│   └── selfplay.yaml     # Self-play data collection
├── play/                  # Play command overrides
│   └── human.yaml        # Human play settings
├── replay/                # Replay command overrides
│   ├── episode.yaml      # Episode replay settings
│   └── browse.yaml       # Episode browsing settings
└── evaluate/              # Evaluation overrides
    └── model.yaml        # Model evaluation settings
```

## Benefits of the New Architecture

### 1. Separation of Concerns
- Base configuration contains all common settings
- Command-specific configs only contain overrides
- Clear organization by functionality

### 2. Flexible Overrides
```bash
# Override any parameter from command line
python main.py train bc model.transformer.embed_dim=128

# Override multiple parameters
python main.py train rl training.rl.total_timesteps=5000000 model.ppo.learning_rate=0.0001

# Override nested parameters
python main.py train bc data_collection.bc_data.output_dir=custom_data
```

### 3. Type Safety and Validation
- Hydra provides automatic type checking
- Configuration validation at startup
- Clear error messages for invalid configs

### 4. Backward Compatibility
- All existing commands still work
- No breaking changes to user workflows
- Gradual adoption of new features

### 5. Improved Debugging
- Hydra shows exactly which configuration is loaded
- Clear trace of where each value comes from
- Easy to reproduce experiments with exact config

## Implementation Phases

### Phase 1: Setup
1. Create config directory structure
2. Move unified_training.yaml to base.yaml
3. Create command-specific override files

### Phase 2: Core Integration
1. Update main.py with new Hydra integration
2. Implement command detection logic
3. Add config composition functionality

### Phase 3: Pipeline Updates
1. Update TrainingPipeline to use DictConfig
2. Update DataCollectionPipeline to use DictConfig
3. Update other pipelines to use DictConfig

### Phase 4: Testing and Documentation
1. Test all commands with new system
2. Verify Hydra overrides work correctly
3. Update documentation with examples

This architecture provides a clean, maintainable, and powerful configuration system while preserving the existing command structure.