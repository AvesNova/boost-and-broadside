"""The unified evaluator (`bnb unified`): one continuously running engine per size bucket
that plays every evaluation target's games at once, refilling each environment the moment
its game ends with the game the planner values most. See docs/rewrite/sections/unified-eval/.
"""
