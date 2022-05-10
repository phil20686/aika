import pandas as pd
import pytest

from aika.time.tests.utils import assert_call
from aika.time.time_range import TimeRange
from aika.time.timestamp import Timestamp

utc_index = pd.Index(
    Timestamp(x) for x in pd.date_range("2022-04-21", freq="H", periods=10)
)
ny_index = pd.Index(
    [
        pd.Timestamp(x, tz="America/New_York")
        for x in pd.date_range("2022-04-21", freq="H", periods=10)
    ]
)


@pytest.mark.parametrize(
    "start, end, expect, str_repr",
    [
        (
            None,
            "2020-01-01T12:00 [America/New_York]",
            TimeRange(None, "2020-01-01T12:00 [America/New_York]"),
            "TimeRange('1677-09-21T00:12:43.145224193 [UTC]', '2020-01-01T12:00:00.0 [America/New_York]')",
        ),
        (
            "2020-01-01T12:00 [UTC]",
            None,
            TimeRange("2020-01-01T12:00 [UTC]", None),
            "TimeRange('2020-01-01T12:00:00.0 [UTC]', '2262-04-11T23:47:16.854775807 [UTC]')",
        ),
        (
            "2020-01-01T12:00 [UTC]",
            "2020-01-01T12:00 [America/New_York]",
            TimeRange("2020-01-01T12:00 [UTC]", "2020-01-01T12:00 [America/New_York]"),
            "TimeRange('2020-01-01T12:00:00.0 [UTC]', '2020-01-01T12:00:00.0 [America/New_York]')",
        ),
        (
            "2020-01-01T12:00 [UTC]",
            "2020-01-01T12:00:00.123456789 [America/New_York]",
            TimeRange(
                "2020-01-01T12:00 [UTC]",
                "2020-01-01T12:00:00.123456789 [America/New_York]",
            ),
            "TimeRange('2020-01-01T12:00:00.0 [UTC]', '2020-01-01T12:00:00.123456789 [America/New_York]')",
        ),
        (
            "2020-01-01T12:00 [UTC]",
            "2020-01-01T12:00:00.1234 [America/New_York]",
            TimeRange(
                "2020-01-01T12:00 [UTC]",
                "2020-01-01T12:00:00.1234000 [America/New_York]",
            ),
            "TimeRange('2020-01-01T12:00:00.0 [UTC]', '2020-01-01T12:00:00.1234 [America/New_York]')",
        ),
        (
            "2020-01-01T12:00 [America/New_York]",
            "2020-01-01T12:00 [UTC]",
            ValueError,
            None,
        ),
    ],
)
def test_constructor_and_equality(start, end, expect, str_repr):
    val = assert_call(TimeRange, expect, start, end)
    if val is not None:
        assert hash(expect) == hash(val)
        assert str_repr == repr(val) == str(val)
        tr2 = eval(repr(val))
        assert val == tr2 == expect


@pytest.mark.parametrize("level", [None, 0])
@pytest.mark.parametrize(
    "time_range, tensor, expect",
    [
        (
            TimeRange("2020-01-01", "2020-01-10"),
            pd.Series(1.0, index=utc_index),
            pd.Series(
                dtype=float,
                index=pd.DatetimeIndex(
                    [],
                    dtype=pd.DatetimeTZDtype.construct_from_string(
                        "datetime64[ns, UTC]"
                    ),
                ),
            ),
        ),
        (
            TimeRange("2022-04-21T00:00", "2022-04-21T02:00:00"),
            pd.Series(1.0, index=utc_index),
            pd.Series(
                1.0,
                dtype=float,
                index=[
                    Timestamp("2022-04-21T01:00:00"),
                    Timestamp("2022-04-21T02:00:00"),
                ],
            ),
        ),
        (
            TimeRange("2022-04-21T10:00", "2022-04-21T12:00:00"),
            pd.Series(1.0, index=ny_index),
            pd.Series(
                1.0,
                dtype=float,
                index=[
                    Timestamp("2022-04-21T07:00:00 [America/New_York]"),
                    Timestamp("2022-04-21T08:00:00 [America/New_York]"),
                ],
            ),
        ),
        (
            TimeRange("2020-01-01", "2020-01-10"),
            pd.DataFrame(1.0, index=utc_index, columns=list("ABC")),
            pd.DataFrame(
                columns=list("ABC"),
                dtype=float,
                index=pd.DatetimeIndex(
                    [],
                    dtype=pd.DatetimeTZDtype.construct_from_string(
                        "datetime64[ns, UTC]"
                    ),
                ),
            ),
        ),
        (
            TimeRange("2022-04-21T00:00", "2022-04-21T02:00:00"),
            pd.DataFrame(1.0, index=utc_index, columns=list("ABC")),
            pd.DataFrame(
                1.0,
                dtype=float,
                index=[
                    Timestamp("2022-04-21T01:00:00"),
                    Timestamp("2022-04-21T02:00:00"),
                ],
                columns=list("ABC"),
            ),
        ),
        (
            TimeRange("2022-04-21T10:00", "2022-04-21T12:00:00"),
            pd.DataFrame(1.0, index=ny_index, columns=list("ABC")),
            pd.DataFrame(
                1.0,
                dtype=float,
                columns=list("ABC"),
                index=[
                    Timestamp("2022-04-21T07:00:00 [America/New_York]"),
                    Timestamp("2022-04-21T08:00:00 [America/New_York]"),
                ],
            ),
        ),
    ],
)
def test_view(time_range, tensor, level, expect):
    assert_call(time_range.view, expect, tensor=tensor, level=level)


base = TimeRange("2000-01-01 00:00", "2000-01-02 00:00")
subset = TimeRange("2000-01-01 10:00", "2000-01-01 12:00")
empty = TimeRange("2000-01-01 00:00", "2000-01-01 00:00")


@pytest.mark.parametrize(
    "first, second, expect",
    [(base, base, True), (base, subset, True), (subset, base, False)],
)
def test_contains(first, second, expect):
    assert first.contains(second) == expect


@pytest.mark.parametrize(
    "time_range, ts, expect",
    [
        (base, base.start, False),
        (base, base.end, True),
        (base, subset.start, True),
    ],
)
def test_in(time_range, ts, expect):
    assert (ts in time_range) == expect
