"""Rendering, installing, checking, and refusing to publish."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from boost_and_broadside.artifacts import ArtifactRecipe
from boost_and_broadside.artifacts.store import file_sha256
from boost_and_broadside.publication.provenance import (
    OWNERSHIP_RELATIVE_PATH,
    PROVENANCE_RELATIVE_PATH,
)
from boost_and_broadside.publication.publish import (
    CHANGED,
    MISSING,
    RENDERED,
    UNCHANGED,
    UNSELECTED,
    run_publish,
)
from boost_and_broadside.publication.renderer_api import (
    PublicationError,
    Renderer,
    register,
)
from tests.publication.conftest import (
    build_artifact,
    fixture_store,
    record_commit,
    write_manifest,
)

_SUMMARY = """
schema_version = 1

[publications.summary]
renderer = "fixture-summary-v1"
output = "docs/results/summary.json"
description = "A fixture aggregate."

[publications.summary.artifacts]
measurement = "{location}"
"""

_REPORT = """
schema_version = 1

[publications.report]
renderer = "fixture-report-v1"
output = "docs/report"
description = "A fixture report tree."

[publications.report.artifacts]
measurement = "{location}"
"""


_TWO_ENTRIES = """
schema_version = 1

[publications.kept]
renderer = "fixture-summary-v1"
output = "docs/results/kept.json"
description = "Still owned."

[publications.kept.artifacts]
measurement = "{location}"

[publications.dropped]
renderer = "fixture-summary-v1"
output = "docs/results/dropped.json"
description = "Dropped from the inventory below."

[publications.dropped.artifacts]
measurement = "{location}"
"""

_ONE_ENTRY = """
schema_version = 1

[publications.kept]
renderer = "fixture-summary-v1"
output = "docs/results/kept.json"
description = "Still owned."

[publications.kept.artifacts]
measurement = "{location}"
"""


def _publish_summary(repository, payload=None, **kwargs):
    location = build_artifact(repository, payload or {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    return run_publish(repository, **kwargs)


def test_a_selected_entry_is_rendered_into_docs(repository, renderers) -> None:
    report = _publish_summary(repository)

    output = repository / "docs" / "results" / "summary.json"
    assert report.by_status(RENDERED)
    assert json.loads(output.read_text()) == {"value": 1}
    assert not report.failed


def test_publishing_twice_reports_the_second_run_as_unchanged(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))

    run_publish(repository)
    report = run_publish(repository)

    assert [outcome.status for outcome in report.outcomes] == [UNCHANGED]


def test_check_passes_on_a_freshly_published_tree_and_writes_nothing(
    repository, renderers
) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    run_publish(repository)
    before = sorted(path.name for path in (repository / "docs" / "results").iterdir())

    report = run_publish(repository, check=True)

    assert not report.failed
    assert [outcome.status for outcome in report.outcomes] == [UNCHANGED]
    assert sorted(path.name for path in (repository / "docs" / "results").iterdir()) == before


def test_check_fails_on_a_missing_output(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))

    report = run_publish(repository, check=True)

    assert report.failed
    assert report.by_status(MISSING)


def test_check_fails_on_a_hand_edited_output(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    run_publish(repository)
    (repository / "docs" / "results" / "summary.json").write_text('{"value": 99}\n')

    report = run_publish(repository, check=True)

    assert report.failed
    assert report.by_status(CHANGED)


def test_a_directory_entry_prunes_what_the_renderer_no_longer_produces(
    repository, renderers
) -> None:
    location = build_artifact(repository, {"sections": ["one", "two"]})
    write_manifest(repository, _REPORT.format(location=location))
    run_publish(repository)
    assert (repository / "docs" / "report" / "two.md").is_file()

    location = build_artifact(repository, {"sections": ["one"]})
    write_manifest(repository, _REPORT.format(location=location))
    report = run_publish(repository)

    assert not (repository / "docs" / "report" / "two.md").exists()
    assert "pruned" in report.outcomes[0].detail


def test_check_reports_a_stale_file_inside_a_directory_entry(repository, renderers) -> None:
    location = build_artifact(repository, {"sections": ["one"]})
    write_manifest(repository, _REPORT.format(location=location))
    run_publish(repository)
    (repository / "docs" / "report" / "left_over.md").write_text("# stale\n")

    report = run_publish(repository, check=True)

    assert report.failed
    assert "stale" in report.outcomes[0].detail
    assert (repository / "docs" / "report" / "left_over.md").is_file()


def test_an_output_the_manifest_no_longer_owns_is_removed(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    run_publish(repository)
    assert (repository / "docs" / "results" / "summary.json").is_file()

    write_manifest(
        repository,
        _SUMMARY.replace("publications.summary", "publications.renamed")
        .replace("results/summary.json", "results/renamed.json")
        .format(location=location),
    )
    report = run_publish(repository)

    assert "docs/results/summary.json" in report.removed
    assert not (repository / "docs" / "results" / "summary.json").exists()


def test_check_fails_on_an_output_the_manifest_no_longer_owns(repository, renderers) -> None:
    # Every entry the manifest still owns is unchanged, so the stale output is
    # the only signal there is: if it does not fail the check, nothing does.
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _TWO_ENTRIES.format(location=location))
    run_publish(repository)
    dropped = repository / "docs" / "results" / "dropped.json"
    assert dropped.is_file()

    write_manifest(repository, _ONE_ENTRY.format(location=location))
    report = run_publish(repository, check=True)

    assert [outcome.status for outcome in report.outcomes] == [UNCHANGED]
    assert report.stale == ["docs/results/dropped.json"]
    assert report.failed
    # Check mode changes nothing, so it may not claim to have removed anything.
    assert report.removed == []
    assert "removed" not in report.render()
    assert dropped.is_file()


def test_publish_removes_what_check_only_reported(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _TWO_ENTRIES.format(location=location))
    run_publish(repository)
    write_manifest(repository, _ONE_ENTRY.format(location=location))
    run_publish(repository, check=True)

    report = run_publish(repository)

    assert report.removed == ["docs/results/dropped.json"]
    assert report.stale == []
    assert not report.failed
    assert not (repository / "docs" / "results" / "dropped.json").exists()
    assert not run_publish(repository, check=True).failed


def test_an_ownership_record_naming_a_path_outside_docs_deletes_nothing(
    repository, renderers
) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    run_publish(repository)
    outsider = repository / "checkpoints" / "keep_me.pt"
    outsider.parent.mkdir(parents=True, exist_ok=True)
    outsider.write_bytes(b"not a canonical output")
    ownership = repository / OWNERSHIP_RELATIVE_PATH
    ownership.write_text(
        json.dumps({"schema_version": 1, "outputs": ["../checkpoints/keep_me.pt"]}) + "\n"
    )

    with pytest.raises(PublicationError, match="must be a path inside docs/"):
        run_publish(repository)
    assert outsider.is_file()


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ("{not json", "not valid JSON"),
        ('{"schema_version": 1}', "list of output paths"),
        ('{"schema_version": 1, "outputs": "docs/results/dropped.json"}', "list of output paths"),
        ('{"schema_version": 1, "outputs": [17]}', "list of output paths"),
    ],
)
def test_an_unreadable_ownership_record_is_an_error_not_an_empty_one(
    repository, renderers, record, message
) -> None:
    # Reading a damaged record as "nothing was owned" would switch the stale
    # check off silently, which is the failure this whole path exists to catch.
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _TWO_ENTRIES.format(location=location))
    run_publish(repository)
    write_manifest(repository, _ONE_ENTRY.format(location=location))
    (repository / OWNERSHIP_RELATIVE_PATH).write_text(record)

    with pytest.raises(PublicationError, match=message):
        run_publish(repository, check=True)


def test_an_absent_ownership_record_means_nothing_was_published_yet(
    repository, renderers
) -> None:
    # Unlike a damaged one, no record at all is the ordinary first-run state.
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    assert not (repository / OWNERSHIP_RELATIVE_PATH).exists()

    report = run_publish(repository)

    assert report.stale == [] and report.removed == []
    assert not report.failed


def test_an_incomplete_source_is_refused(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1}, complete=False)
    write_manifest(repository, _SUMMARY.format(location=location))

    with pytest.raises(PublicationError, match="in-progress"):
        run_publish(repository)
    assert not (repository / "docs" / "results" / "summary.json").exists()


def test_an_interrupted_resumable_sweep_is_refused_as_a_source(repository, renderers) -> None:
    # A resumable sweep rewrites result.json after every batch, so an interrupted
    # one leaves a payload that verifies perfectly and reports a fraction of the
    # games it was asked for. Only the recorded status distinguishes it.
    store = fixture_store(repository)
    recipe = ArtifactRecipe("fixture", 1, subjects={"fixture": True}, parameters={"games": 100})
    artifact, resumed = store.open_resumable(recipe, store.standalone_owner())
    assert not resumed
    artifact.write_json({"value": 1, "games": 50})
    artifact, resumed = store.open_resumable(recipe, store.standalone_owner())
    assert resumed
    artifact.write_json({"value": 1, "games": 75})  # killed before complete()
    record_commit(artifact.path)
    location = artifact.path.relative_to(repository)
    write_manifest(repository, _SUMMARY.format(location=location))

    with pytest.raises(PublicationError, match="cannot be cited"):
        run_publish(repository)

    # Finishing the same measurement makes it citable, and nothing else changes.
    artifact.complete()
    record_commit(artifact.path)
    report = run_publish(repository)
    assert report.by_status(RENDERED)
    assert json.loads(
        (repository / "docs" / "results" / "summary.json").read_text()
    ) == {"value": 1, "games": 75}


def test_an_unselected_entry_is_reported_and_left_alone(repository, renderers) -> None:
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

    report = run_publish(repository, check=True)

    assert [outcome.status for outcome in report.outcomes] == [UNSELECTED]
    assert not report.failed
    assert not (repository / "docs" / "results").exists()


def test_targeting_an_unselected_entry_is_a_loud_error(repository, renderers) -> None:
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

    with pytest.raises(PublicationError, match="cannot be rendered yet"):
        run_publish(repository, target="summary")


def test_a_dirty_source_is_refused(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1}, clean=False)
    write_manifest(repository, _SUMMARY.format(location=location))

    with pytest.raises(PublicationError, match="dirty checkout"):
        run_publish(repository)


def test_a_tampered_source_payload_is_refused(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    (repository / location / "result.json").write_text('{"value": 2}')

    with pytest.raises(PublicationError, match="recorded hash"):
        run_publish(repository)


def test_a_source_schema_the_renderer_cannot_read_is_refused(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1}, schema_version=7)
    write_manifest(repository, _SUMMARY.format(location=location))

    with pytest.raises(PublicationError, match="result schema"):
        run_publish(repository)


def test_promoted_media_must_match_its_recorded_hash(repository, renderers) -> None:
    clip = repository / "out" / "clip.gif"
    clip.parent.mkdir()
    clip.write_bytes(b"GIF89a-fixture")
    body = """
        schema_version = 1

        [publications.clip]
        renderer = "fixture-media-v1"
        output = "docs/results/replays/clip.gif"
        description = "A promoted clip."

        [publications.clip.files]
        clip = {{ path = "out/clip.gif", sha256 = "{digest}" }}
        """
    write_manifest(repository, body.format(digest=file_sha256(clip)))

    report = run_publish(repository)
    published = repository / "docs" / "results" / "replays" / "clip.gif"
    assert published.read_bytes() == clip.read_bytes()
    assert report.by_status(RENDERED)

    clip.write_bytes(b"GIF89a-changed")
    with pytest.raises(PublicationError, match="recorded sha256"):
        run_publish(repository)


def test_a_renderer_that_reaches_the_network_fails(repository, isolated_registry) -> None:
    def connect(inputs, out_dir: Path) -> list[Path]:
        socket.create_connection(("example.invalid", 80))
        return []

    register(
        Renderer(
            name="fixture-network-v1",
            description="Deliberately misbehaving.",
            render=connect,
            required_artifacts=("measurement",),
        )
    )
    location = build_artifact(repository, {"value": 1})
    write_manifest(
        repository,
        _SUMMARY.replace("fixture-summary-v1", "fixture-network-v1").format(location=location),
    )

    real_socket = socket.socket
    with pytest.raises(PublicationError, match="offline"):
        run_publish(repository)
    # The guard has to put the process back as it found it, even on failure.
    assert socket.socket is real_socket


def test_a_single_file_renderer_may_not_produce_several(repository, isolated_registry) -> None:
    register(
        Renderer(
            name="fixture-greedy-v1",
            description="Writes more than it owns.",
            render=lambda inputs, out_dir: [
                (out_dir / name).write_text("x") or (out_dir / name) for name in ("a.txt", "b.txt")
            ],
            required_artifacts=("measurement",),
        )
    )
    location = build_artifact(repository, {"value": 1})
    write_manifest(
        repository,
        _SUMMARY.replace("fixture-summary-v1", "fixture-greedy-v1").format(location=location),
    )

    with pytest.raises(PublicationError, match="owns one file"):
        run_publish(repository)


def test_a_failed_install_leaves_the_previous_output_intact(
    repository, renderers, monkeypatch
) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    run_publish(repository)
    output = repository / "docs" / "results" / "summary.json"

    location = build_artifact(repository, {"value": 2})
    write_manifest(repository, _SUMMARY.format(location=location))
    monkeypatch.setattr(
        "boost_and_broadside.publication.publish.os.replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError):
        run_publish(repository)
    assert json.loads(output.read_text()) == {"value": 1}


def test_the_generated_index_names_each_source_and_only_appears_once_selected(
    repository, renderers
) -> None:
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
    run_publish(repository)
    assert not (repository / PROVENANCE_RELATIVE_PATH).exists()
    assert not (repository / OWNERSHIP_RELATIVE_PATH).exists()

    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    run_publish(repository)

    index = (repository / PROVENANCE_RELATIVE_PATH).read_text()
    assert "Do not edit" in index
    assert "fixture-summary-v1" in index
    assert str(location) in index
    ownership = json.loads((repository / OWNERSHIP_RELATIVE_PATH).read_text())
    assert ownership["outputs"] == ["docs/results/summary.json"]


def test_publish_never_writes_outside_docs(repository, renderers) -> None:
    location = build_artifact(repository, {"value": 1})
    write_manifest(repository, _SUMMARY.format(location=location))
    before = {
        path.relative_to(repository)
        for path in repository.rglob("*")
        if path.is_file() and "docs" not in path.parts
    }

    run_publish(repository)

    after = {
        path.relative_to(repository)
        for path in repository.rglob("*")
        if path.is_file() and "docs" not in path.parts
    }
    assert after == before
