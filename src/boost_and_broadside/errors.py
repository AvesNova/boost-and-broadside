"""The one exception base the CLI reports without a traceback.

A failure that is the user's to fix — a missing artifact, an invalid manifest, a
payload that no longer matches its hash — should read as one line, while an
internal failure should still surface its traceback. Library code raises these;
``bnb`` translates them and nothing else.

This module deliberately imports nothing, so the lightweight parser can catch
them without pulling in Torch.
"""

from __future__ import annotations


class UserFacingError(RuntimeError):
    """A failure to report as a concise message instead of a traceback."""


__all__ = ["UserFacingError"]
