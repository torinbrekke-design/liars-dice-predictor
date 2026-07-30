import math

import pytest

from prediction_models.games import liars_dice as ld


# --- primitives ----------------------------------------------------------- #
def test_match_probability_wild_vs_ones():
    assert math.isclose(ld.match_probability(3), 2 / 6)   # face or wild 1
    assert math.isclose(ld.match_probability(1), 1 / 6)   # ones: only literal 1s
    assert math.isclose(ld.match_probability(3, ones_wild=False), 1 / 6)


def test_count_matches_counts_face_and_wilds():
    hand = [1, 3, 3, 5, 6]
    assert ld.count_matches(hand, 3) == 3       # two 3s + one wild 1
    assert ld.count_matches(hand, 6) == 2       # one 6 + one wild 1
    assert ld.count_matches(hand, 2) == 1       # just the wild 1
    assert ld.count_matches(hand, 1) == 1       # ones: only the literal 1


# --- assessment ----------------------------------------------------------- #
def test_bid_already_guaranteed_true():
    # Hand alone meets the bid -> probability 1 regardless of unknowns.
    a = ld.assess_bid(total_dice=25, bid_quantity=3, bid_face=3, hand=[1, 3, 3, 5, 6])
    assert a.my_matches == 3
    assert a.needed_from_unknown == 0
    assert math.isclose(a.p_at_least, 1.0)
    assert math.isclose(a.probability, 1.0)


def test_needs_one_from_unknown():
    # Need exactly 1 of 20 unknown dice at p=1/3: P>=1 = 1 - (2/3)^20.
    a = ld.assess_bid(total_dice=25, bid_quantity=4, bid_face=3, hand=[1, 3, 3, 5, 6])
    assert a.unknown_dice == 20
    assert a.needed_from_unknown == 1
    expected = 1 - (2 / 3) ** 20
    assert math.isclose(a.p_at_least, expected, rel_tol=1e-9)


def test_red_die_bonus_is_one_short_times_one_sixth():
    a = ld.assess_bid(total_dice=10, bid_quantity=6, bid_face=4, hand=[2, 2, 5, 6, 3])
    assert math.isclose(a.red_die_bonus, a.p_one_short / 6.0, rel_tol=1e-12)
    assert math.isclose(a.probability, a.p_at_least + a.red_die_bonus, rel_tol=1e-12)
    # Including the red die can only help.
    no_red = ld.bid_probability(10, 6, 4, [2, 2, 5, 6, 3], use_red_die=False)
    assert a.probability >= no_red


def test_impossible_bid_is_zero():
    # Need more matches than there are unknown dice (and none in hand).
    a = ld.assess_bid(total_dice=6, bid_quantity=20, bid_face=4, hand=[2, 3, 5, 6, 6])
    assert a.p_at_least == 0.0
    assert a.probability == 0.0  # also too far short for the red die


def test_ones_bid_uses_one_sixth():
    a = ld.assess_bid(total_dice=12, bid_quantity=3, bid_face=1, hand=[1, 2, 3, 4, 5])
    assert math.isclose(a.match_prob, 1 / 6)
    assert a.my_matches == 1  # only the literal 1, no wilds on a 1-bid


# --- exact vs simulation -------------------------------------------------- #
@pytest.mark.parametrize("q,face", [(4, 3), (6, 4), (8, 5), (3, 1)])
def test_exact_matches_simulation(q, face):
    hand = [1, 3, 3, 5, 6]
    exact = ld.bid_probability(20, q, face, hand)
    sim = ld.simulate_bid(20, q, face, hand, trials=60_000, seed=123)
    assert abs(exact - sim.probability) < 0.01


# --- calling "liar" ------------------------------------------------------- #
def test_call_is_complement_of_bid_standing():
    c = ld.assess_call(25, 12, 3, [1, 3, 3, 5, 6])
    assert math.isclose(c.prob_bid_false, 1.0 - c.prob_bid_stands, rel_tol=1e-12)
    assert math.isclose(
        c.expected_die_swing, c.prob_bid_false - c.prob_bid_stands, rel_tol=1e-12
    )


def test_call_recommended_when_bid_unlikely():
    # A wild over-bid: should clearly be a call.
    c = ld.assess_call(25, 18, 3, [2, 2, 2, 5, 6])
    assert c.prob_bid_false > 0.5
    assert c.should_call is True


def test_call_not_recommended_when_bid_safe():
    # A modest bid you can nearly cover yourself: don't call.
    c = ld.assess_call(25, 4, 3, [1, 3, 3, 5, 6])
    assert c.prob_bid_stands > 0.5
    assert c.should_call is False


def test_call_threshold_is_respected():
    c = ld.assess_call(25, 10, 3, [1, 3, 3, 5, 6], threshold=0.9)
    # Bid stands ~55%, so false ~45% < 90% -> do not call at this threshold.
    assert c.should_call is False


# --- validation ----------------------------------------------------------- #
def test_validation_errors():
    with pytest.raises(ValueError):
        ld.assess_bid(total_dice=0, bid_quantity=1, bid_face=3, hand=[3])
    with pytest.raises(ValueError):
        ld.assess_bid(total_dice=10, bid_quantity=0, bid_face=3, hand=[3])
    with pytest.raises(ValueError):
        ld.assess_bid(total_dice=10, bid_quantity=1, bid_face=7, hand=[3])
    with pytest.raises(ValueError):
        ld.assess_bid(total_dice=2, bid_quantity=1, bid_face=3, hand=[3, 3, 3])
    with pytest.raises(ValueError):
        ld.assess_bid(total_dice=10, bid_quantity=1, bid_face=3, hand=[])
