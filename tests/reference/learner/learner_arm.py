"""The learner fixture interface over bnb (see README.md).

The learner section implements this adapter: `make_arm()` returns an object with every
method README.md lists, each a thin translation onto `bnb.learn`. Until then it returns
None and the equivalence tests skip.
"""


def make_arm():
    return None
