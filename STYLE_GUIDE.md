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
    3.  Local application imports (e.g., `src.env...`).
    *   *Tip*: Use `isort` or `ruff` to manage this automatically.
*   **Whitespace**:
    *   Two blank lines between top-level definitions (classes, functions).
    *   One blank line between methods inside a class.

## 3. Naming Conventions

| Type | Convention | Example |
| :--- | :--- | :--- |
| **Modules** | `snake_case` | `train_model.py`, `utils.py` |
| **Classes** | `PascalCase` | `WorldModel`, `TeamTransformerAgent` |
| **Functions/Methods** | `snake_case` | `compute_loss()`, `get_active_positions()` |
| **Variables** | `snake_case` | `batch_size`, `num_ships` |
| **Constants** | `UPPER_CASE` | `MAX_SHIPS`, `DEFAULT_TIMEOUT` |

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
*   **Grouping**: Group related files by feature (e.g., `src/env`, `src/agents`), but do not subdivide further unless absolutely necessary.

### 6.2. Function & Method Complexity
*   **Single Responsibility**: A function should do one thing and do it well. If a function performs multiple distinct tasks, split it.
*   **Length**: Prefer small functions. While there is no hard limit, functions longer than **40-50 lines** are often a sign that they are doing too much and should be refactored.
*   **Nesting**: Maximum **3 levels** of indentation.
    *   *Solution*: Use "guard clauses" (early returns) to reduce nesting.
    *   *Solution*: Extract inner loops or complex logic into helper methods.

### 6.3. Configuration (Hydra)
*   **No Defaults in Code**: Do not provide default values in Python code for parameters that should be controlled by config files.
    *   *Bad*: `learning_rate = cfg.get("lr", 0.001)`
    *   *Good*: `learning_rate = cfg.lr` (Let it crash if `lr` is missing).
*   **Validation**: Validate config values early (e.g., at the start of `main()` or `__init__`).

### 6.4. Classes vs. Functions
*   **Use Classes** when you need to maintain state or bundle data with behavior.
*   **Use Functions** for pure logic, transformations, or simple scripts.
*   **Module Organization**: Prefer focused modules with clear boundaries.
    *   Large, complex classes should generally live in their own file (e.g., `WorldModel` in `world_model.py`).
    *   Related classes may be grouped in a single file when they form a cohesive unit (e.g., exception hierarchies, protocol + implementation, small related variants).
    *   Helper classes (dataclasses, configs, small utilities) can live alongside their primary class.
    *   *Guideline*: If a file grows beyond **300-400 lines** or contains unrelated classes, consider splitting it.
*   **Inheritance**: Prefer **composition over inheritance**.
    *   Inherit from `nn.Module` for PyTorch models.
    *   Avoid deep inheritance chains (A -> B -> C -> D).

### 6.5. Entry Point
*   The only valid entry point is `uv run main.py`.
*   All other scripts in `src/` should be modules, not executable scripts (unless they are specific utility scripts in `tools/`).

### 6.6. Backward Compatibility
*   **No Backward Compatibility**: We do not maintain support for deprecated APIs, old patterns, or legacy code.
*   **Remove, Don't Deprecate**: When refactoring, delete old code entirely rather than marking it as deprecated.
    *   *Rationale*: Deprecated code clutters the codebase, creates maintenance burden, and encourages continued use of outdated patterns.
*   **Breaking Changes**: Breaking changes are acceptable and preferred if they improve code quality, readability, or maintainability.

## 7. Comments
*   **Inline Comments**: Use them to explain **why**, not **what**.
*   **Block Comments**: Use them to explain complex algorithms or sections of code.
*   **Variable Names**: Prefer renaming variables to be self-explanatory over adding comments.