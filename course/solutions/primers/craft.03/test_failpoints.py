"""My tests for the craft.03 kata (the reference learner's artifact).

Written before the code, one failing test at a time (red), then the code
that makes it pass (green). They import only the spec's names, so they
grade any implementation of it.
"""

import pytest

from failpoints import Failpoints, Rule, parse, sleep_seconds


def test_entry_with_count_and_whitespace():
    assert parse(" train/after-step = 3 * crash ") == {"train/after-step": Rule("crash", "", 3)}


def test_empty_entries_are_skipped():
    assert parse("a=off;;b=panic;") == {"a": Rule("off", "", 0), "b": Rule("panic", "", 0)}
    assert parse("") == {}


def test_error_message_is_kept_and_trimmed():
    assert parse("e=error( disk full )")["e"] == Rule("error", "disk full", 0)
    assert parse("e=error")["e"] == Rule("error", "", 0)


def test_nth_fires_once_on_the_nth_evaluation():
    fp = Failpoints("x=2*crash")
    assert [fp.evaluate("x") is not None for _ in range(4)] == [False, True, False, False]


def test_without_count_fires_every_time():
    fp = Failpoints("x=panic")
    assert all(fp.evaluate("x") == Rule("panic", "", 0) for _ in range(3))


def test_off_never_fires_or_counts():
    fp = Failpoints("x=off")
    assert fp.evaluate("x") is None and fp.count("x") == 0


def test_unknown_name_is_not_counted():
    fp = Failpoints("x=crash")
    assert fp.evaluate("y") is None and fp.count("y") == 0 and fp.count("x") == 0


def test_counts_are_per_name():
    fp = Failpoints("a=2*crash;b=2*crash")
    fp.evaluate("a")
    assert fp.evaluate("b") is None
    assert fp.evaluate("a") is not None


def test_sleep_units():
    assert sleep_seconds("250ms") == 0.25
    assert sleep_seconds("3s") == 3.0
    assert parse("s=sleep(10ms)")["s"].arg == "10ms"


@pytest.mark.parametrize(
    "spec",
    ["a", "=crash", "a=boom", "a=0*crash", "a=crash;a=panic", "a=off(x)", "a=sleep", "a=sleep(5)", "a=b=crash"],
)
def test_malformed_specs_are_errors(spec):
    with pytest.raises(ValueError):
        parse(spec)
