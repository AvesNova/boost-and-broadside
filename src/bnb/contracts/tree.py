"""Contract dataclasses as pytrees, so the runtime can map copy_, clone and to() over them."""

import dataclasses

from torch.utils import _pytree


def contract[T: type](cls: T) -> T:
    """Make `cls` a frozen dataclass and register it as a pytree (its fields are the leaves).

    A field may hold a tensor, None, or another contract.
    """
    cls = dataclasses.dataclass(frozen=True)(cls)
    _pytree.register_dataclass(cls)
    return cls
