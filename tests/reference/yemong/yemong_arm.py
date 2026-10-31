"""The Yemong fixture interface over bnb's model code (see README.md).

The Yemong section implements this adapter: `make_arm()` returns an object with every
method README.md lists, each a thin translation onto `bnb` (fixture weight names to module
parameters, the neutral carry layout to the model's own). Until then it returns None and
the equivalence tests skip.
"""


def make_arm():
    return None
