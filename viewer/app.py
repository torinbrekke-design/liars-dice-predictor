"""Liar's Dice predictor — a small Streamlit input window.

Run it with:
    streamlit run viewer/app.py

It imports the same tested `prediction_models.games.liars_dice` engine the CLI
and test suite use, so the numbers here always match.
"""

from __future__ import annotations

import streamlit as st

from prediction_models.games import liars_dice as ld

st.set_page_config(page_title="Liar's Dice Predictor", page_icon="🎲", layout="centered")

st.title("🎲 Liar's Dice Predictor")
st.caption(
    "1s are wild (except when the bid itself is on 1s). If a challenged bid is "
    "exactly one short, the red-die last chance (1/6) is folded in."
)


def parse_hand(text: str) -> list[int]:
    parts = text.replace(",", " ").split()
    hand = [int(p) for p in parts]
    for face in hand:
        if not 1 <= face <= ld.SIDES:
            raise ValueError(f"die '{face}' is out of range (1-6)")
    if not hand:
        raise ValueError("enter at least one die")
    return hand


# --- inputs --------------------------------------------------------------- #
with st.sidebar:
    st.header("Your situation")
    total = st.number_input(
        "Dice left on the table (including yours)",
        min_value=1, max_value=60, value=25, step=1,
    )
    hand_text = st.text_input("Your hand (faces, e.g. 1 3 3 5 6)", value="1 3 3 5 6")

    st.divider()
    st.header("The bid")
    bid_qty = st.number_input("Quantity", min_value=1, max_value=60, value=8, step=1)
    bid_face = st.selectbox("Face value", options=[1, 2, 3, 4, 5, 6], index=2)

    use_red = st.checkbox("Include red-die 'within 1' rule", value=True)

# --- validate ------------------------------------------------------------- #
try:
    hand = parse_hand(hand_text)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if len(hand) > total:
    st.error(f"Your hand has {len(hand)} dice but the table total is only {total}.")
    st.stop()

# --- compute (same engine as the CLI/tests) ------------------------------- #
a = ld.assess_bid(total, bid_qty, bid_face, hand, use_red_die=use_red)
c = ld.assess_call(total, bid_qty, bid_face, hand, use_red_die=use_red)

st.subheader(f"Bid:  {bid_qty} × {bid_face}'s")

col1, col2 = st.columns(2)
col1.metric("Chance the bid STANDS", f"{a.probability * 100:.1f}%")
col2.metric("Chance the bid is FALSE", f"{c.prob_bid_false * 100:.1f}%")

st.progress(min(a.probability, 1.0))

if c.should_call:
    st.success(
        f"✅ **CALL liar** — the bid is more likely false "
        f"({c.prob_bid_false * 100:.1f}%) than true."
    )
else:
    st.info(
        f"🚫 **Don't call** — the bid likely stands "
        f"({a.probability * 100:.1f}%). Hold or raise."
    )

# --- breakdown ------------------------------------------------------------ #
with st.expander("Show the full breakdown", expanded=True):
    st.markdown(
        f"""
- **Unknown dice** (everyone but you): **{a.unknown_dice}**
- **Guaranteed matches in your hand**: **{a.my_matches}**
  _(your {bid_face}'s{' + wild 1s' if bid_face != 1 else ''})_
- **Still needed from the unknown dice**: **{max(a.needed_from_unknown, 0)}**
  &nbsp;(each unknown die matches with p = {a.match_prob:.3f})
- **P(bid literally true, count ≥ {bid_qty})**: **{a.p_at_least * 100:.1f}%**
- **P(exactly one short)**: {a.p_one_short * 100:.1f}%
  → red die saves **+{a.red_die_bonus * 100:.1f}%**
- **Headline P(bid stands)**: **{a.probability * 100:.1f}%**
"""
    )

st.caption(
    "Model: each unknown die independently matches the bid face (or a wild 1) — "
    "an exact binomial tail. Reuses the tested liars_dice engine."
)
