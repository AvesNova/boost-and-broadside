"""Every public module, class and function has a docstring (code style, Docstrings)."""

from . import docstrings

# Owned by open branches (the play shell's second milestone, fix/intermittent), which write
# their own; the cleanup pass left them alone. Remove an entry when its branch merges.
EXEMPT = ("bnb.play", "bnb.models.yemong.kernels")


def test_every_public_object_has_a_docstring():
    missing = [
        f"{kind} {name}" for kind, name in docstrings.missing() if not name.startswith(EXEMPT)
    ]
    assert not missing, "\n".join(missing)
