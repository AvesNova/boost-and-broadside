"""``key=value`` overrides applied to a profile before resolution.

One positional argument per changed value::

    bnb train --profile rl clip_coef=0.2 elo_eval.window_size=64

The form is positional rather than a flag per hyperparameter because the point
is a sweep: W&B's ``${args_no_hyphens}`` emits exactly this, so an arm's config
reaches the trainer without a per-parameter argparse entry that has to be added
whenever a new value becomes interesting.

Values are coerced against the dataclass field they land on, so ``clip_coef=0.2``
arrives as a float and ``league_uniform_sampling=true`` as a bool. An unknown key
is an error naming the closest real ones: a silently ignored override in a sweep
produces an arm that claims to have tested something it did not.
"""

from __future__ import annotations

import dataclasses
from difflib import get_close_matches
from functools import cache
from typing import Any, get_args, get_origin, get_type_hints

from boost_and_broadside.errors import UserFacingError

_TRUE = {"true", "yes", "1", "on"}
_FALSE = {"false", "no", "0", "off"}


class OverrideError(UserFacingError):
    """An override does not name a real field, or does not fit the one it names."""


@cache
def _hints(cls: type) -> dict[str, Any]:
    """Real annotations for a dataclass.

    ``field.type`` is the *source text* of the annotation under
    ``from __future__ import annotations``, so coercing against it would compare
    a value to the string "float".
    """

    return get_type_hints(cls)


def _leaf_names(config: Any, prefix: str = "") -> list[str]:
    names = []
    for field in dataclasses.fields(config):
        path = f"{prefix}{field.name}"
        value = getattr(config, field.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            names.extend(_leaf_names(value, f"{path}."))
        else:
            names.append(path)
    return names


def _coerce(text: str, annotation: Any, path: str) -> Any:
    if text == "none" and _optional(annotation):
        return None
    target = _concrete(annotation)
    try:
        if target is bool:
            lowered = text.lower()
            if lowered in _TRUE:
                return True
            if lowered in _FALSE:
                return False
            raise ValueError(f"expected a boolean, got {text!r}")
        if target is int:
            # Accept 1e6 and 1_000_000 for step counts and budgets.
            return int(float(text)) if {"e", "E", "."} & set(text) else int(text, 0)
        if target is float:
            return float(text)
        if target is str:
            return text
    except ValueError as error:
        raise OverrideError(f"{path}={text!r}: {error}") from error
    raise OverrideError(f"{path} is a {annotation}, which cannot be set from the command line")


def _optional(annotation: Any) -> bool:
    return get_origin(annotation) is not None and type(None) in get_args(annotation)


def _concrete(annotation: Any) -> Any:
    if get_origin(annotation) is None:
        return annotation
    concrete = [arg for arg in get_args(annotation) if arg is not type(None)]
    return concrete[0] if len(concrete) == 1 else annotation


def parse_override(text: str) -> tuple[str, str]:
    """Split one ``key=value`` argument, or explain why it is not one."""

    key, separator, value = text.partition("=")
    if not separator or not key.strip():
        raise OverrideError(f"expected key=value, got {text!r}")
    return key.strip(), value.strip()


def apply_overrides(config: Any, overrides: dict[str, str]) -> Any:
    """Return ``config`` with each dotted path replaced, coerced to its field type."""

    result = config
    for path, text in overrides.items():
        result = _apply_one(result, path, path.split("."), text)
    return result


def _apply_one(config: Any, path: str, parts: list[str], text: str) -> Any:
    head, *rest = parts
    names = {field.name: field for field in dataclasses.fields(config)}
    if head not in names:
        available = _leaf_names(config)
        suggestion = get_close_matches(path, available, n=3)
        detail = f"; did you mean {', '.join(suggestion)}?" if suggestion else ""
        raise OverrideError(f"unknown config key {path!r}{detail}")
    if rest:
        child = getattr(config, head)
        if not dataclasses.is_dataclass(child):
            raise OverrideError(f"{head} is not a group, so {path!r} has nowhere to go")
        return dataclasses.replace(config, **{head: _apply_one(child, path, rest, text)})
    return dataclasses.replace(
        config, **{head: _coerce(text, _hints(type(config))[head], path)}
    )
