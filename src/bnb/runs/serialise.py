"""Frozen dataclasses to JSON and back; pytrees of tensors to named tensors and back (runs
spec 2).

The JSON round trip is exact: floats are written with `repr` (the `json` module's own
behaviour), tuples come back as tuples and enums by name. Reading is driven by the class's
type hints, so a value of the wrong kind, an unknown field or a missing field without a
default fails with the field's dotted path. A missing field with a default takes it, and
the read reports the paths it filled (the model-version rule, runs spec 4).
"""

import dataclasses
import enum
import types
import typing
from typing import Any

import torch
from torch.utils import _pytree


class SerialiseError(ValueError):
    """A stored value that does not fit the type it is read as."""


def to_json(value: Any) -> Any:
    """`value` as JSON data: a dataclass as a dict of its fields, tuples and lists as lists,
    an enum by its name. Numbers, strings, booleans and None are themselves."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, tuple | list):
        return [to_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise SerialiseError(f"{type(value).__name__} is not JSON data")


def from_json[T](cls: type[T], data: Any) -> tuple[T, list[str]]:
    """(the value of type `cls` that `to_json` wrote as `data`, the dotted paths of the
    fields a missing entry filled with their defaults)."""
    filled: list[str] = []
    return _read(cls, data, "", filled), filled


def _read(kind: Any, data: Any, path: str, filled: list[str]) -> Any:
    where = path or "the value"
    origin, arguments = typing.get_origin(kind), typing.get_args(kind)
    if origin in (typing.Union, types.UnionType):
        if data is None and type(None) in arguments:
            return None
        options = [option for option in arguments if option is not type(None)]
        errors = []
        for option in options:
            try:
                return _read(option, data, path, filled)
            except SerialiseError as error:
                errors.append(str(error))
        raise SerialiseError("; ".join(errors))
    if dataclasses.is_dataclass(kind):
        return _read_dataclass(kind, data, path, filled)
    if origin is tuple:
        if not isinstance(data, list):
            raise SerialiseError(f"{where}: expected a list, got {data!r}")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _read(arguments[0], item, f"{path}[{index}]", filled)
                for index, item in enumerate(data)
            )
        if len(arguments) != len(data):
            raise SerialiseError(f"{where}: expected {len(arguments)} items, got {len(data)}")
        return tuple(
            _read(argument, item, f"{path}[{index}]", filled)
            for index, (argument, item) in enumerate(zip(arguments, data))
        )
    if origin is list:
        if not isinstance(data, list):
            raise SerialiseError(f"{where}: expected a list, got {data!r}")
        return [
            _read(arguments[0], item, f"{path}[{index}]", filled) for index, item in enumerate(data)
        ]
    if origin is dict:
        if not isinstance(data, dict):
            raise SerialiseError(f"{where}: expected an object, got {data!r}")
        return {
            key: _read(arguments[1], item, f"{path}.{key}", filled) for key, item in data.items()
        }
    if isinstance(kind, type) and issubclass(kind, enum.Enum):
        if data not in kind.__members__:
            raise SerialiseError(f"{where}: {data!r} is not one of {list(kind.__members__)}")
        return kind[data]
    if kind is float:
        if isinstance(data, bool) or not isinstance(data, int | float):
            raise SerialiseError(f"{where}: expected a number, got {data!r}")
        return float(data)
    if kind in (bool, int, str):
        if type(data) is not kind:
            raise SerialiseError(f"{where}: expected {kind.__name__}, got {data!r}")
        return data
    if kind is Any:
        return data
    raise SerialiseError(f"{where}: cannot read a {kind!r}")


def _read_dataclass(cls: type, data: Any, path: str, filled: list[str]) -> Any:
    if not isinstance(data, dict):
        raise SerialiseError(f"{path or cls.__name__}: expected an object, got {data!r}")
    hints = typing.get_type_hints(cls)
    fields = {field.name: field for field in dataclasses.fields(cls) if field.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        names = ", ".join(f"{path}.{name}" if path else name for name in unknown)
        raise SerialiseError(f"unknown field {names} in {cls.__name__}")
    values = {}
    for name, field in fields.items():
        dotted = f"{path}.{name}" if path else name
        if name in data:
            values[name] = _read(hints[name], data[name], dotted, filled)
        elif (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        ):
            filled.append(dotted)
        else:
            raise SerialiseError(f"{cls.__name__} lacks {dotted}, which has no default")
    return cls(**values)


# ---------------------------------------------------------------- named tensors


def _key_name(entry: Any) -> str:
    if isinstance(entry, _pytree.MappingKey):
        return str(entry.key)
    if isinstance(entry, _pytree.GetAttrKey):
        return entry.name
    if isinstance(entry, _pytree.SequenceKey):
        return str(entry.idx)
    raise SerialiseError(f"unsupported pytree key {entry!r}")


def named_tensors(tree: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    """Every tensor leaf of `tree` by its key path ("state.ships.x"), under `prefix`."""
    leaves, _ = _pytree.tree_flatten_with_path(tree)
    named = {}
    for key_path, leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            name = ".".join(_key_name(entry) for entry in key_path)
            named[f"{prefix}{name}" if prefix else name] = leaf
    return named


def tree_from_named(target: Any, named: dict[str, torch.Tensor], prefix: str = "") -> Any:
    """A tree shaped like `target` whose tensor leaves are `named`'s (under `prefix`), each
    on its target leaf's device. Every name, shape and dtype must match the target's, and
    no stored name under the prefix may be left over."""
    wanted = named_tensors(target, prefix)
    stored = {name for name in named if name.startswith(prefix)}
    missing, extra = sorted(set(wanted) - stored), sorted(stored - set(wanted))
    if missing or extra:
        raise SerialiseError(
            f"stored tensors do not match: missing {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}, unexpected {extra[:5]}"
            f"{' ...' if len(extra) > 5 else ''}"
        )
    for name, leaf in wanted.items():
        value = named[name]
        if value.shape != leaf.shape or value.dtype != leaf.dtype:
            raise SerialiseError(
                f"{name}: stored {tuple(value.shape)} {value.dtype}, "
                f"expected {tuple(leaf.shape)} {leaf.dtype}"
            )
    leaves, spec = _pytree.tree_flatten(target)
    names = iter(wanted)
    loaded = [
        named[next(names)].to(leaf.device) if isinstance(leaf, torch.Tensor) else leaf
        for leaf in leaves
    ]
    return _pytree.tree_unflatten(loaded, spec)
