"""Manifest validation: what the inventory is allowed to say."""

from __future__ import annotations

import pytest

from boost_and_broadside.publication.manifest import ManifestError, load_manifest
from boost_and_broadside.publication.renderer_api import PublicationError
from tests.publication.conftest import build_artifact, write_manifest

_SELECTED = """
schema_version = 1

[publications.summary]
renderer = "fixture-summary-v1"
output = "docs/results/summary.json"
description = "A fixture aggregate."

[publications.summary.artifacts]
measurement = "{location}"
"""


def test_a_complete_entry_is_selected(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SELECTED.format(location=location))

    manifest = load_manifest(repository)

    assert [entry.name for entry in manifest.entries] == ["summary"]
    assert manifest.selected == manifest.entries
    assert manifest.entries[0].output == "docs/results/summary.json"


def test_an_entry_without_its_required_source_is_unselected(repository, renderers) -> None:
    write_manifest(
        repository,
        """
        schema_version = 1

        [publications.summary]
        renderer = "fixture-summary-v1"
        output = "docs/results/summary.json"
        description = "Not chosen yet."
        """,
    )

    entry = load_manifest(repository).entries[0]

    assert not entry.selected
    assert entry.missing_sources == ("measurement",)
    assert load_manifest(repository).selected == ()


def test_a_renderer_with_no_required_source_is_always_selected(repository, isolated_registry):
    from boost_and_broadside.publication.renderer_api import Renderer, register

    register(
        Renderer(
            name="fixture-static-v1",
            description="An asset with no producer here.",
            render=lambda inputs, out_dir: [],
        )
    )
    write_manifest(
        repository,
        """
        schema_version = 1

        [publications.diagram]
        renderer = "fixture-static-v1"
        output = "docs/diagram.png"
        description = "Not produced in this repository."
        """,
    )

    assert load_manifest(repository).entries[0].selected


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("schema_version = 2\n[publications.a]\n", "manifest schema"),
        ("schema_version = 1\n", "declares no publications"),
        (
            'schema_version = 1\nextra = 1\n[publications.a]\nrenderer = "fixture-summary-v1"\n'
            'output = "docs/a.json"\ndescription = "x"\n',
            "unknown top-level keys",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "no-such-renderer"\n'
            'output = "docs/a.json"\ndescription = "x"\n',
            "unknown renderer",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "fixture-summary-v1"\n'
            'output = "results/a.json"\ndescription = "x"\n',
            "escapes docs",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "fixture-summary-v1"\n'
            'output = "docs/../a.json"\ndescription = "x"\n',
            "inside docs",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "fixture-summary-v1"\n'
            'output = "/etc/a.json"\ndescription = "x"\n',
            "inside docs",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "fixture-summary-v1"\n'
            'output = "docs/a.json"\ndescription = "x"\nmystery = 1\n',
            "unknown keys",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "fixture-summary-v1"\n'
            'output = "docs/a.json"\ndescription = "x"\n'
            '[publications.a.artifacts]\nwrong = "artifacts/x"\n',
            "unknown artifact source",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "fixture-media-v1"\n'
            'output = "docs/a.gif"\ndescription = "x"\n'
            '[publications.a.files]\nclip = { path = "out/a.gif" }\n',
            "exactly path and sha256",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "fixture-media-v1"\n'
            'output = "docs/a.gif"\ndescription = "x"\n'
            '[publications.a.files]\nclip = { path = "out/a.gif", sha256 = "abc" }\n',
            "64-character sha256",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "fixture-media-v1"\n'
            'output = "docs/a.gif"\ndescription = "x"\n'
            f'[publications.a.files]\nclip = {{ path = "out/a.gif", sha256 = "{"z" * 64}" }}\n',
            "not hexadecimal",
        ),
        (
            'schema_version = 1\n[publications.a]\nrenderer = "fixture-summary-v1"\n'
            'output = "docs/a.json"\ndescription = "x"\n'
            '[publications.b]\nrenderer = "fixture-summary-v1"\n'
            'output = "docs/a.json"\ndescription = "y"\n',
            "both own",
        ),
        ("this is not toml =", "not valid TOML"),
    ],
)
def test_invalid_manifests_are_rejected_with_the_reason(repository, renderers, body, message):
    write_manifest(repository, body)

    with pytest.raises(PublicationError, match=message):
        load_manifest(repository)


def test_a_missing_manifest_names_the_path_it_looked_for(repository) -> None:
    with pytest.raises(ManifestError, match="publications.toml"):
        load_manifest(repository)


def test_an_unknown_publication_lists_what_the_manifest_declares(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SELECTED.format(location=location))

    with pytest.raises(ManifestError, match="summary"):
        load_manifest(repository).entry("absent")
