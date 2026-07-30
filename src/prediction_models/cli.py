"""Table-side interactive CLI for the Liar's Dice prediction model.

Run it with the installed console script::

    liars-dice

or directly::

    python -m prediction_models.cli

Flow: enter the total dice on the table and your hand once per round, then type
bids to evaluate. Commands:

    8 3            assess making/holding the bid "8 threes"
    call 8 3       assess calling "liar" on the bid "8 threes"
    hand           re-enter your hand (e.g. after a re-roll)
    total          re-enter the table's total dice count
    new            start a fresh round (re-enter total + hand)
    help           show commands
    q / quit       exit
"""

from __future__ import annotations

from prediction_models.games import liars_dice as ld

PROMPT = (
    "\nBid to check  ['Q F' | 'call Q F' | hand | total | new | help | q]: "
)


def _parse_dice(text: str) -> list[int]:
    """Parse a hand like '1 3 3 5 6' or '1,3,3,5,6' into a list of faces."""
    parts = text.replace(",", " ").split()
    hand = [int(p) for p in parts]
    for face in hand:
        if not 1 <= face <= ld.SIDES:
            raise ValueError(f"die face {face} out of range 1..{ld.SIDES}")
    if not hand:
        raise ValueError("no dice entered")
    return hand


def _ask_int(prompt: str, reader) -> int:
    while True:
        try:
            value = int(reader(prompt).strip())
            if value < 1:
                print("  must be >= 1")
                continue
            return value
        except (ValueError, TypeError):
            print("  please enter a whole number")


def _ask_hand(reader) -> list[int]:
    while True:
        try:
            return _parse_dice(reader("Your hand (e.g. 1 3 3 5 6): "))
        except ValueError as exc:
            print(f"  {exc}")


def _parse_bid(tokens: list[str]) -> tuple[int, int]:
    if len(tokens) != 2:
        raise ValueError("expected two numbers: quantity and face")
    quantity, face = int(tokens[0]), int(tokens[1])
    if quantity < 1:
        raise ValueError("quantity must be >= 1")
    if not 1 <= face <= ld.SIDES:
        raise ValueError(f"face must be 1..{ld.SIDES}")
    return quantity, face


def run(reader=input) -> None:
    """Run the interactive loop. ``reader`` is injectable for testing."""
    print("=== Liar's Dice prediction model ===")
    print("(1s are wild except on 1-bids; 'within 1' triggers the red die)")

    total = _ask_int("Total dice on the table (including yours): ", reader)
    hand = _ask_hand(reader)
    if len(hand) > total:
        print("  note: hand larger than total; bumping total to match")
        total = len(hand)

    while True:
        try:
            raw = reader(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        cmd, *rest = raw.split()
        low = cmd.lower()

        if low in ("q", "quit", "exit"):
            break
        if low == "help":
            print(__doc__)
            continue
        if low == "total":
            total = _ask_int("Total dice on the table (including yours): ", reader)
            continue
        if low == "hand":
            hand = _ask_hand(reader)
            continue
        if low == "new":
            total = _ask_int("Total dice on the table (including yours): ", reader)
            hand = _ask_hand(reader)
            continue

        try:
            if low == "call":
                quantity, face = _parse_bid(rest)
                print(ld.assess_call(total, quantity, face, hand).summary())
            else:
                quantity, face = _parse_bid([cmd, *rest])
                print(ld.assess_bid(total, quantity, face, hand).summary())
        except ValueError as exc:
            print(f"  {exc}  (type 'help' for usage)")


def main() -> None:
    try:
        run()
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
