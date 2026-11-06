"""The league fixture interface over bnb (see README.md).

`make_arm()` returns an object with every method README.md lists, each a thin translation
onto `bnb.league.ratings`.
"""

import types

from bnb.league import ratings


def make_arm():
    return types.SimpleNamespace(
        fit_ratings=ratings.fit_ratings,
    )
