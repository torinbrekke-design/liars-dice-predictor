"""Liar's Dice prediction model.

House rules this models (4-6 players, 5 d6 each to start, hidden in cups):

* 1s ("aces") are WILD and count as the current bid's face value — UNLESS the
  bid itself is on the 1 face, in which case only literal 1s count.
* A bid is "Q of face F" (e.g. 4 threes). It is *literally true* when the total
  number of matching dice across the whole table is >= Q. A tie (exactly Q)
  favors the bidder: a challenger who calls "liar" loses.
* "Within 1" red-die last chance: if a bidder is challenged and the table is
  exactly ONE short of the bid (count == Q-1), the bidder rolls a single red d6
  in the middle. If it shows the bid's face F exactly (1s are NOT wild on this
  roll, so probability 1/6), that die counts and the bid stands — the challenger
  loses instead. Being 2+ short means no red-die chance.

The model is exact (binomial), since the unknown dice are i.i.d. A Monte-Carlo
cross-check (:func:`simulate_bid`) is provided for confidence.

Core question answered: *given the total dice in play, my own hand, and a bid,
what is the probability the bid stands?*
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb

from prediction_models.core import MonteCarloResult, monte_carlo

__all__ = [
    "SIDES",
    "WILD_FACE",
    "RED_DIE_HIT_PROB",
    "BidAssessment",
    "CallAssessment",
    "match_probability",
    "count_matches",
    "assess_bid",
    "bid_probability",
    "assess_call",
    "simulate_bid",
]

SIDES = 6
WILD_FACE = 1
RED_DIE_HIT_PROB = 1.0 / SIDES  # ones are NOT wild on the red die


# --------------------------------------------------------------------------- #
# Probability primitives
# --------------------------------------------------------------------------- #
def match_probability(bid_face: int, ones_wild: bool = True) -> float:
    """Probability that one unknown die counts toward a bid on ``bid_face``.

    With wild ones, any non-ace bid is matched by the face itself OR a 1
    (2/6 = 1/3). A bid on the 1 face is matched only by a literal 1 (1/6).
    """
    _check_face(bid_face)
    if bid_face == WILD_FACE or not ones_wild:
        return 1.0 / SIDES
    return 2.0 / SIDES


def count_matches(hand: list[int], bid_face: int, ones_wild: bool = True) -> int:
    """Count guaranteed matches in your own hand for a bid on ``bid_face``."""
    _check_face(bid_face)
    for die in hand:
        _check_face(die)
    if bid_face == WILD_FACE or not ones_wild:
        return sum(1 for d in hand if d == bid_face)
    return sum(1 for d in hand if d == bid_face or d == WILD_FACE)


def _binom_pmf(n: int, i: int, p: float) -> float:
    if i < 0 or i > n:
        return 0.0
    return comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))


def _binom_at_least(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(_binom_pmf(n, i, p) for i in range(k, n + 1))


# --------------------------------------------------------------------------- #
# Assessment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BidAssessment:
    """Full breakdown of how likely a bid is to stand.

    Attributes:
        bid_quantity: Q, the claimed number of matching dice.
        bid_face: F, the claimed face value (1..6).
        total_dice: total dice in play across the whole table (includes yours).
        unknown_dice: dice you cannot see (total minus your hand).
        my_matches: guaranteed matches in your own hand (wild 1s included).
        needed_from_unknown: k = Q - my_matches (matches still required).
        match_prob: per-unknown-die probability of matching the bid.
        p_at_least: P(total count >= Q) — the bid is *literally* true.
        p_one_short: P(total count == Q - 1) — eligible for the red die.
        red_die_bonus: p_one_short * 1/6, the extra survival from the red die.
        probability: p_at_least + red_die_bonus — probability the bid STANDS.
    """

    bid_quantity: int
    bid_face: int
    total_dice: int
    unknown_dice: int
    my_matches: int
    needed_from_unknown: int
    match_prob: float
    p_at_least: float
    p_one_short: float
    red_die_bonus: float
    probability: float

    def summary(self) -> str:
        pct = lambda x: f"{x * 100:.1f}%"
        return "\n".join([
            f"Bid: {self.bid_quantity} x {self.bid_face}'s  "
            f"(table has {self.total_dice} dice, {self.unknown_dice} unknown)",
            f"  In your hand           : {self.my_matches} guaranteed match(es)",
            f"  Still needed (unknown) : {max(self.needed_from_unknown, 0)} "
            f"(each unknown die matches with p={self.match_prob:.3f})",
            f"  P(bid literally true)  : {pct(self.p_at_least)}",
            f"  P(exactly 1 short)     : {pct(self.p_one_short)}"
            f"  ->  red-die saves +{pct(self.red_die_bonus)}",
            f"  P(bid STANDS)          : {pct(self.probability)}",
        ])


def assess_bid(
    total_dice: int,
    bid_quantity: int,
    bid_face: int,
    hand: list[int],
    *,
    ones_wild: bool = True,
    use_red_die: bool = True,
) -> BidAssessment:
    """Assess whether a bid stands, given the table size and your own hand.

    Args:
        total_dice: total dice in play across the table, INCLUDING your own.
        bid_quantity: Q — the bid's claimed count.
        bid_face: F — the bid's claimed face (1..6).
        hand: your own dice as a list of faces, e.g. ``[1, 3, 3, 5, 6]``.
        ones_wild: whether 1s are wild (house rule: yes, except on 1-bids).
        use_red_die: include the "within 1" red-die last-chance in the headline
            probability.

    Returns:
        A :class:`BidAssessment` with the full breakdown.
    """
    if total_dice < 1:
        raise ValueError("total_dice must be >= 1")
    if bid_quantity < 1:
        raise ValueError("bid_quantity must be >= 1")
    _check_face(bid_face)
    if not hand:
        raise ValueError("hand must contain at least one die")
    if len(hand) > total_dice:
        raise ValueError("hand has more dice than the table total")

    unknown = total_dice - len(hand)
    p = match_probability(bid_face, ones_wild)
    m = count_matches(hand, bid_face, ones_wild)
    k = bid_quantity - m  # matches still needed from the unknown dice

    p_at_least = _binom_at_least(unknown, k, p)
    p_one_short = _binom_pmf(unknown, k - 1, p) if k - 1 >= 0 else 0.0
    red_die_bonus = p_one_short * RED_DIE_HIT_PROB if use_red_die else 0.0
    probability = min(1.0, p_at_least + red_die_bonus)

    return BidAssessment(
        bid_quantity=bid_quantity,
        bid_face=bid_face,
        total_dice=total_dice,
        unknown_dice=unknown,
        my_matches=m,
        needed_from_unknown=k,
        match_prob=p,
        p_at_least=p_at_least,
        p_one_short=p_one_short,
        red_die_bonus=red_die_bonus,
        probability=probability,
    )


def bid_probability(
    total_dice: int,
    bid_quantity: int,
    bid_face: int,
    hand: list[int],
    *,
    ones_wild: bool = True,
    use_red_die: bool = True,
) -> float:
    """Shortcut: probability the bid stands (the headline number)."""
    return assess_bid(
        total_dice, bid_quantity, bid_face, hand,
        ones_wild=ones_wild, use_red_die=use_red_die,
    ).probability


# --------------------------------------------------------------------------- #
# Calling "liar"
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CallAssessment:
    """Whether to call "liar" on the previous player's bid.

    Reasoned from your seat: every die you can't see (including the bidder's) is
    unknown. Calling forces a reveal; exactly one player loses a die.

    Attributes:
        prob_bid_stands: P(the bid stands) — if you call, YOU lose a die.
        prob_bid_false: P(the bid is false) — if you call, the BIDDER loses a die.
        expected_die_swing: prob_bid_false - prob_bid_stands. Positive favors
            calling (the opponent is more likely than you to lose the die).
        should_call: True when prob_bid_false exceeds ``threshold``.
        threshold: the decision threshold used (default 0.5).
        bid: the underlying :class:`BidAssessment` (full probability breakdown).
    """

    bid_quantity: int
    bid_face: int
    total_dice: int
    unknown_dice: int
    prob_bid_stands: float
    prob_bid_false: float
    expected_die_swing: float
    should_call: bool
    threshold: float
    bid: BidAssessment

    def summary(self) -> str:
        pct = lambda x: f"{x * 100:.1f}%"
        verdict = "CALL liar" if self.should_call else "DO NOT call (let it ride / raise)"
        return "\n".join([
            f"Calling on bid: {self.bid_quantity} x {self.bid_face}'s  "
            f"(table {self.total_dice} dice, {self.unknown_dice} unknown)",
            f"  P(bid is false -> bidder loses) : {pct(self.prob_bid_false)}",
            f"  P(bid stands   -> you lose)     : {pct(self.prob_bid_stands)}",
            f"  Expected die swing (yours+)     : {self.expected_die_swing:+.3f}",
            f"  Verdict (threshold {pct(self.threshold)}) : {verdict}",
        ])


def assess_call(
    total_dice: int,
    bid_quantity: int,
    bid_face: int,
    hand: list[int],
    *,
    ones_wild: bool = True,
    use_red_die: bool = True,
    threshold: float = 0.5,
) -> CallAssessment:
    """Assess calling "liar" on a bid, given the table size and your own hand.

    Args mirror :func:`assess_bid`. ``threshold`` is the minimum P(bid false)
    at which calling is recommended (default 0.5 = call when the bid is more
    likely false than true). The red-die rule helps the bidder, so it is
    included in P(bid stands) by default.
    """
    bid = assess_bid(
        total_dice, bid_quantity, bid_face, hand,
        ones_wild=ones_wild, use_red_die=use_red_die,
    )
    prob_stands = bid.probability
    prob_false = 1.0 - prob_stands
    return CallAssessment(
        bid_quantity=bid_quantity,
        bid_face=bid_face,
        total_dice=total_dice,
        unknown_dice=bid.unknown_dice,
        prob_bid_stands=prob_stands,
        prob_bid_false=prob_false,
        expected_die_swing=prob_false - prob_stands,
        should_call=prob_false > threshold,
        threshold=threshold,
        bid=bid,
    )


# --------------------------------------------------------------------------- #
# Monte-Carlo cross-check
# --------------------------------------------------------------------------- #
def simulate_bid(
    total_dice: int,
    bid_quantity: int,
    bid_face: int,
    hand: list[int],
    *,
    ones_wild: bool = True,
    use_red_die: bool = True,
    trials: int = 50_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Estimate P(bid stands) by simulation — a sanity check on :func:`assess_bid`."""
    _check_face(bid_face)
    unknown = total_dice - len(hand)
    m = count_matches(hand, bid_face, ones_wild)
    counts_one_only = (bid_face == WILD_FACE or not ones_wild)

    def trial(rng: random.Random) -> bool:
        hits = 0
        for _ in range(unknown):
            face = rng.randint(1, SIDES)
            if counts_one_only:
                hits += face == bid_face
            else:
                hits += face == bid_face or face == WILD_FACE
        count = m + hits
        if count >= bid_quantity:
            return True
        if use_red_die and count == bid_quantity - 1:
            return rng.randint(1, SIDES) == bid_face  # ones not wild here
        return False

    return monte_carlo(trial, trials=trials, seed=seed)


def _check_face(face: int) -> None:
    if not (isinstance(face, int) and 1 <= face <= SIDES):
        raise ValueError(f"die face must be an integer in 1..{SIDES}, got {face!r}")
