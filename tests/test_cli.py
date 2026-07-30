from prediction_models import cli


def _scripted_reader(lines):
    """Return a fake input() that yields queued lines, then raises EOFError."""
    it = iter(lines)

    def reader(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return reader


def test_cli_assesses_bid_and_call(capsys):
    reader = _scripted_reader([
        "25",            # total dice
        "1 3 3 5 6",     # hand
        "8 3",           # assess a bid
        "call 18 3",     # assess a call
        "q",             # quit
    ])
    cli.run(reader=reader)
    out = capsys.readouterr().out
    assert "P(bid STANDS)" in out
    assert "CALL liar" in out


def test_cli_handles_bad_input_then_recovers(capsys):
    reader = _scripted_reader([
        "25",
        "1 3 3 5 6",
        "banana",        # invalid bid -> error, keep going
        "8 3",
        "q",
    ])
    cli.run(reader=reader)
    out = capsys.readouterr().out
    assert "P(bid STANDS)" in out


def test_cli_comma_separated_hand(capsys):
    reader = _scripted_reader(["10", "1,3,3,5,6", "4 3", "q"])
    cli.run(reader=reader)
    out = capsys.readouterr().out
    assert "guaranteed match" in out
