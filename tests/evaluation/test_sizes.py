"""Typed fleet-size parsing contracts."""

import pytest

from boost_and_broadside.evaluation.sizes import (
    DEFAULT_MATCHUP,
    Matchup,
    MatchupParseError,
    parse_matchup,
    parse_matchups,
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("4", Matchup(4, 4)),
        ("4v4", Matchup(4, 4)),
        ("3v5", Matchup(3, 5)),
        ("12v2", Matchup(12, 2)),
    ],
)
def test_matchup_parser_preserves_symmetric_and_asymmetric_sides(spec, expected):
    parsed = parse_matchup(spec)
    assert parsed == expected
    assert parsed.num_ships == expected.team0 + expected.team1


@pytest.mark.parametrize(
    "spec",
    ["", "0", "0v4", "4v0", "-1", "+4", "4V4", "4v", "v4", "4v4v4", " 4v4"],
)
def test_matchup_parser_rejects_non_positive_malformed_or_ambiguous_values(spec):
    with pytest.raises(MatchupParseError):
        parse_matchup(spec)


def test_omitted_matchup_list_uses_the_locked_4v4_default():
    assert parse_matchups(None) == [DEFAULT_MATCHUP]
    assert DEFAULT_MATCHUP == Matchup(4, 4)
