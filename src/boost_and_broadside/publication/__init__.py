"""Offline publication: the only writer of canonical views under ``docs/``.

Compute modes write artifacts. ``bnb publish`` reads the manifest, verifies the
artifacts it selects, renders the declared outputs in a temporary directory with
no network access, and installs them atomically. It never simulates, never
plays, and never contacts a service.
"""

from boost_and_broadside.publication.renderer_api import (
    PublicationError,
    Renderer,
    RenderInputs,
    get_renderer,
    register,
    registered_renderers,
)

__all__ = [
    "PublicationError",
    "RenderInputs",
    "Renderer",
    "get_renderer",
    "register",
    "registered_renderers",
]
