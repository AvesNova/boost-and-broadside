"""Every public module, class and function has a docstring (code style, Docstrings)."""

from . import docstrings

# Owned by an open branch (fix/intermittent), which writes its own; the cleanup pass left it
# alone. Remove the entry when its branch merges.
EXEMPT = ("bnb.models.yemong.kernels",)


def test_every_public_object_has_a_docstring():
    missing = [
        f"{kind} {name}" for kind, name in docstrings.missing() if not name.startswith(EXEMPT)
    ]
    assert not missing, "\n".join(missing)
