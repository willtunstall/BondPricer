from monte_carlo_bond_pricer import MonteCarloBondPricer, present_value, main
import pytest

def test_present_value():

    # ((1000 * 0.05) / (1 + 0.025)) + ... + ((1000 * 0.05) / (1 + 0.025)**3) + (1000 / (1 + 0.025)**3) = 1071.4006
    assert int(present_value(1000, 0.05, 0.025, 3, 1)) == 1071

    # ZERO COUPON: ((1000 * 0) / (1 + 0.025)) + ... + ((1000 * 0) / (1 + 0.025)**3) + (1000 / (1 + 0.025)**3) = 928.5994
    assert int(present_value(1000, 0, 0.025, 3, 1)) == 928

def test_simulation():
    ...
    
