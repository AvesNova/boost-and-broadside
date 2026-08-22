"""Every chart a finished run can produce, and the registry that names them.

Compute modes write artifacts; ``bnb figures`` turns one run's artifacts into
that run's charts, in that run's own directory. Nothing here simulates, plays,
or contacts a service, and nothing here writes outside the run it was given.

There is deliberately no second copy anywhere. The documents link at the run
they cite, so a chart in ``docs/`` and the same chart in the run's evidence are
one file rather than two renders that agree by convention.
"""

from boost_and_broadside.charts.renderer_api import (
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
