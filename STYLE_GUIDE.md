# Project Style Guide

This document defines the coding standards for the **Boost and Broadside** project. It is adapted from the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), with specific overrides to suit our project's needs and modern Python practices.

## 1. General Principles

*   **Be Consistent**: If you edit existing code, take a few minutes to look at the code around you and determine its style. If they use spaces around all their arithmetic operators, you should too.
*   **Fail Fast**: Errors should be caught as early as possible. Missing configurations or invalid states should cause an immediate crash rather than silent defaults.
*   **Modern Python**: We use Python 3.13+. Utilize modern features like pattern matching (`match`/`case`) and new type hinting syntax.

## 2. Code Layout & Formatting

*   **Line Length**: Maximum **100 characters**.
*   **Indentation**: **4 spaces**. No tabs.
*   **Imports**:
    1.  Standard library imports.
    2.  Third-party application imports (e.g., `numpy`, `torch`).
    3.  Local application imports (e.g., `boost_and_broadside.env...`).
    *   *Tip*: Use `isort` or `ruff` to manage this automatically.
*   **Whitespace**:
    *   Two blank lines between top-level definitions (classes, functions).
    *   One blank line between methods inside a class.

## 3. Naming Conventions

| Type | Convention | Example |
| :--- | :--- | :--- |
| **Modules** | `snake_case` | `train_model.py`, `utils.py` |
| **Classes** | `PascalCase` | `MVPPolicy`, `TensorEnv` |
| **Functions/Methods** | `snake_case` | `compute_loss()`, `get_active_positions()` |
| **Variables** | `snake_case` | `batch_size`, `num_ships` |
| **Constants** | `UPPER_CASE` | `TOTAL_ACTION_LOGITS`, `NUM_POWER_ACTIONS` |

*   **Avoid**: Single-letter names (except `i`, `j`, `k` in loops, or `x`, `y` in math contexts).
*   **Descriptive Names**: Variable names should be descriptive enough that comments are often unnecessary to explain *what* the variable holds.

## 4. Typing & Type Hints

We strictly enforce type hints for all function signatures.

*   **Modern Syntax**: Use Python 3.10+ syntax.
    *   **Union**: Use `X | Y` instead of `Union[X, Y]` or `Optional[X]`.
        *   *Good*: `def get_item(id: int) -> Item | None:`
        *   *Bad*: `def get_item(id: int) -> Optional[Item]:`
    *   **Generics**: Use built-in collections.
        *   *Good*: `list[int]`, `dict[str, Any]`
        *   *Bad*: `List[int]`, `Dict[str, Any]`
*   **Coverage**: All arguments and return values must be typed.

## 5. Documentation (Docstrings)

We follow the **Google Style** for docstrings.

*   **Modules**: Start with a docstring explaining the module's purpose.
*   **Classes**: Class docstring should explain the class's purpose and public attributes.
*   **Functions**:
    *   **Args**: List each argument, its type (if not obvious), and description.
    *   **Returns**: Describe the return value.
    *   **Raises**: List exceptions that are explicitly raised.

```python
def calculate_velocity(distance: float, time: float) -> float:
    """Calculates velocity based on distance and time.

    Args:
        distance: The distance traveled in meters.
        time: The time taken in seconds.

    Returns:
        The velocity in meters per second.

    Raises:
        ValueError: If time is zero or negative.
    """
    if time <= 0:
        raise ValueError("Time must be positive.")
    return distance / time
```

## 6. Project-Specific Guidelines

### 6.1. Folder Structure
*   **Shallow Nesting**: Keep the directory structure flat. Avoid creating deep hierarchies (e.g., `src/a/b/c/d.py`).
*   **Grouping**: Group related files by feature (e.g., `src/env`, `src/models`), but do not subdivide further unless absolutely necessary.

### 6.2. Function & Method Complexity
*   **Single Responsibility**: A function should do one thing and do it well. If a function performs multiple distinct tasks, split it.
*   **Length**: Prefer small functions. Functions longer than **40-50 lines** are often a sign they are doing too much. **Exception**: Vectorized GPU kernels (e.g., physics updates, attention computations) may be longer when splitting would force redundant tensor allocations or destroy locality. Comment these with a `# GPU kernel: kept together for performance` note.
*   **Nesting**: Maximum **3 levels** of indentation.
    *   *Solution*: Use "guard clauses" (early returns) to reduce nesting.
    *   *Solution*: Extract inner loops or complex logic into helper methods.

### 6.3. Configuration (Frozen Dataclasses)

We use `@dataclass(frozen=True)` for all configuration. No Hydra, no OmegaConf.

*   **No Magic Numbers**: Do not hardcode numeric constants inline. Every constant that controls behavior must be in a config dataclass or `constants.py`.
    *   *Bad*: `reward = damage * 0.001`
    *   *Good*: `reward = damage * config.damage_weight`
*   **No Defaults for Hyperparameters**: Training, model, and reward configs must have **no default field values**. Every value must be explicitly specified at the call site in `main.py`.
    *   *Bad*: `learning_rate: float = 3e-4`
    *   *Good*: `learning_rate: float  # required — set in main.py`
*   **Physics defaults OK**: `ShipConfig` (physics simulation parameters) may have default values since these define the game model, not a tunable hyperparameter.
*   **Immutability enforced**: `frozen=True` means mutations raise `FrozenInstanceError` at runtime, catching config bugs early.
*   **Validation**: Validate config values immediately after construction (e.g., assert `n_heads` divides `d_model`).

```python
# Good
@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float   # no default
    num_envs: int          # no default
    gamma: float           # no default

# main.py — all values explicit
cfg = TrainConfig(learning_rate=3e-4, num_envs=64, gamma=0.99)
```

### 6.4. Tensor Shape Annotations

Every non-trivial tensor must have a shape comment on the line it is created or transformed. Use the `(dim0, dim1, ...)` format with named dimensions.

*   Use single letters for common dims: `B` = batch/num_envs, `N` = num_ships, `T` = time/steps, `D` = feature dim, `H` = heads, `K` = bullets.
*   Put the comment inline for short lines, above for complex reshapes.

```python
x = self.encoder(obs)                              # (B, N, D)
x = x.unsqueeze(1)                                 # (B, 1, N, D) — add time dim
x = x.reshape(B * N, T, self.d_model)              # (B*N, T, D) — flatten for GRU
attn_mask = alive.unsqueeze(1).unsqueeze(2)        # (B, 1, 1, N) — broadcast over heads
```

### 6.5. GPU / PyTorch Rules

*   **No `.item()`, `.cpu()`, `.numpy()` inside training loops**. These cause synchronization barriers that destroy GPU throughput. Only permitted in:
    1.  The async logging path (off the hot path)
    2.  Test assertions
*   **Pre-allocate tensors**: Never `torch.zeros(...)` inside a step loop. Allocate buffers in `__init__` or `reset()`.
*   **Avoid Python loops over batch dims**: Use tensor operations. If you find yourself writing `for i in range(num_envs):`, that is a red flag.
*   **Device discipline**: Every tensor must be on the correct device from creation. Pass `device` explicitly. Never rely on implicit CPU fallback.
*   **In-place ops**: Prefer in-place tensor writes (e.g., `state.ship_health[mask] = value`) for large buffers to avoid allocations, but be careful with autograd.

### 6.6. Classes vs. Functions
*   **Use Classes** when you need to maintain state or bundle data with behavior.
*   **Use Functions** for pure logic, transformations, or simple scripts.
*   **Module Organization**: Prefer focused modules with clear boundaries.
    *   Large, complex classes should generally live in their own file.
    *   Related classes may be grouped in a single file when they form a cohesive unit.
    *   Helper classes (dataclasses, small utilities) can live alongside their primary class.
    *   *Guideline*: If a file grows beyond **300-400 lines** or contains unrelated classes, consider splitting it.
*   **Inheritance**: Prefer **composition over inheritance**.
    *   Inherit from `nn.Module` for PyTorch models.
    *   Avoid deep inheritance chains (A -> B -> C -> D).

### 6.7. Entry Point
*   The only valid entry point is `uv run --no-sync main.py`.
*   **Note for AI Agents**: Always use `uv run --no-sync` instead of `uv run` when executing Python commands or scripts to avoid background execution issues.
*   All other scripts in `src/` should be modules, not executable scripts (unless they are specific utility scripts in `tools/`).

### 6.8. Backward Compatibility
*   **No Backward Compatibility**: We do not maintain support for deprecated APIs, old patterns, or legacy code.
*   **Remove, Don't Deprecate**: When refactoring, delete old code entirely rather than marking it as deprecated.
    *   *Rationale*: Deprecated code clutters the codebase, creates maintenance burden, and encourages continued use of outdated patterns.
*   **Breaking Changes**: Breaking changes are acceptable and preferred if they improve code quality, readability, or maintainability.

### 6.9. Testing Philosophy
*   **One logical assertion per test**: Each test function should verify exactly one behavior. Use descriptive test names that read as specifications.
    *   *Good*: `test_thrust_increases_velocity_in_attitude_direction`
    *   *Bad*: `test_physics` (tests 12 things)
*   **Physical invariants make the best tests**: Test conservation laws, symmetries, and physical constraints (e.g., "boosting always increases speed", "dead ships have zero health") rather than exact floating-point values.
*   **No mocking physics**: Do not mock the physics engine or GPU tensors in env tests. Run against real tensors (CPU is fine in tests). We got burned before when mocked tests passed but the real environment had silent bugs.
*   **Test at system boundaries**: Test the public API of each module (what goes in, what comes out). Do not test private helpers directly unless they have complex, isolated logic.
*   **Parametrize over configurations**: Use `@pytest.mark.parametrize` to test the same behavior across multiple configs or inputs rather than copy-pasting test bodies.

## 7. Comments
*   **Inline Comments**: Use them to explain **why**, not **what**.
*   **Block Comments**: Use them to explain complex algorithms or sections of code.
*   **Variable Names**: Prefer renaming variables to be self-explanatory over adding comments.
