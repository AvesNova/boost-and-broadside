"""Construction seams that must not acquire a default.

``team_pma_k`` decides which value heads are team-pooled. A default here would be
silently wrong rather than loudly missing, which is why it is checked at the
signature rather than at a call site.

Three ``ast``-walking tests were removed from this file: they enforced import
direction and a no-``sys.exit`` rule by parsing every module in the package. Both
rules still hold, but a syntax-tree scan is a slow, brittle way to state them --
it re-derives the whole package on every run to check something that shows up as
an ordinary failure the moment it is violated.
"""

import inspect

from boost_and_broadside.models.yemong.policy import YemongPolicy
from boost_and_broadside.train.rl.policy_io import build_policy
from boost_and_broadside.train.rl.roster import EloRoster


def test_team_pma_indices_are_required_at_policy_and_roster_construction_seams():
    policy_parameter = inspect.signature(YemongPolicy).parameters["team_pma_k"]
    builder_parameter = inspect.signature(build_policy).parameters["team_pma_k"]
    assert policy_parameter.default is inspect.Parameter.empty
    assert builder_parameter.default is inspect.Parameter.empty
    roster_parameter = inspect.signature(EloRoster.load_policy).parameters["team_pma_k"]
    assert roster_parameter.default is inspect.Parameter.empty
