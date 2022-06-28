import typing as t

import pandas as pd
import pytest

from aika.utilities.pandas_utils import IndexTensor, Tensor
from aika.utilities.testing import assert_call, assert_equal

from aika.time.time_range import RESOLUTION, TimeRange
from aika.time.timestamp import Timestamp


def timestamp_index(*args, tz=None, **kwargs):
    if tz is None:
        tz = "UTC"
    return pd.date_range(*args, **kwargs, tz=tz)


empty_ts_index = pd.DatetimeIndex([], dtype=pd.DatetimeTZDtype(tz="UTC"))


@pytest.mark.parametrize(
    "start, end, expect, str_repr",
    [
        (
            None,
            "2020-01-01T12:00 [America/New_York]",
            TimeRange(None, "2020-01-01T12:00 [America/New_York]"),
            "TimeRange('1677-09-21T00:12:43.145224193 [UTC]', '2020-01-01T12:00:00 [America/New_York]')",
        ),
        (
            "2021-01-01T12:00",
            Timestamp("2021-01-01T12:00") + RESOLUTION,
            TimeRange("2021-01-01T12:00", "2021-01-01T12:00:00.000000001"),
            "TimeRange('2021-01-01T12:00:00 [UTC]', '2021-01-01T12:00:00.000000001 [UTC]')",
        ),
        (
            "2020-01-01T12:00 [UTC]",
            None,
            TimeRange("2020-01-01T12:00 [UTC]", None),
            "TimeRange('2020-01-01T12:00:00 [UTC]', '2262-04-11T23:47:16.854775807 [UTC]')",
        ),
        (
            "2020-01-01T12:00 [UTC]",
            "2020-01-01T12:00 [America/New_York]",
            TimeRange("2020-01-01T12:00 [UTC]", "2020-01-01T12:00 [America/New_York]"),
            "TimeRange('2020-01-01T12:00:00 [UTC]', '2020-01-01T12:00:00 [America/New_York]')",
        ),
        (
            "2020-01-01T12:00 [UTC]",
            "2020-01-01T12:00:00.123456789 [America/New_York]",
            TimeRange(
                "2020-01-01T12:00 [UTC]",
                "2020-01-01T12:00:00.123456789 [America/New_York]",
            ),
            "TimeRange('2020-01-01T12:00:00 [UTC]', '2020-01-01T12:00:00.123456789 [America/New_York]')",
        ),
        (
            "2020-01-01T12:00 [UTC]",
            "2020-01-01T12:00:00.1234 [America/New_York]",
            TimeRange(
                "2020-01-01T12:00 [UTC]",
                "2020-01-01T12:00:00.1234000 [America/New_York]",
            ),
            "TimeRange('2020-01-01T12:00:00 [UTC]', '2020-01-01T12:00:00.123400 [America/New_York]')",
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


@pytest.mark.parametrize(
    "time_range, tensor, expect",
    [
        (
            TimeRange("2020-01-01", "2020-01-10"),
            pd.Series(
                1.0, index=timestamp_index(start="2022-04-21", freq="H", periods=10)
            ),
            pd.Series(
                dtype=float,
                index=empty_ts_index,
            ),
        ),
        (
            TimeRange("2022-04-21T00:00", "2022-04-21T02:00:00"),
            pd.Series(
                1.0, index=timestamp_index(start="2022-04-21", freq="H", periods=10)
            ),
            pd.Series(
                1.0,
                dtype=float,
                index=[
                    Timestamp("2022-04-21T00:00:00"),
                    Timestamp("2022-04-21T01:00:00"),
                ],
            ),
        ),
        (
            TimeRange("2022-04-21T10:00", "2022-04-21T12:00:00"),
            pd.Series(
                1.0,
                index=timestamp_index(
                    start="2022-04-21", freq="H", periods=10, tz="America/New_York"
                ),
            ),
            pd.Series(
                1.0,
                dtype=float,
                index=[
                    Timestamp("2022-04-21T06:00:00 [America/New_York]"),
                    Timestamp("2022-04-21T07:00:00 [America/New_York]"),
                ],
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "tensor_type",
    [
        pytest.param(lambda s: s, id="Series"),
        pytest.param(lambda s: pd.DataFrame(dict(A=s, B=s + 1, C=-s)), id="DataFrame"),
        pytest.param(lambda s: s.index, id="Index"),
    ],
)
@pytest.mark.parametrize(
    "index_transform, level",
    [
        pytest.param(
            lambda ix: ix,
            0,
            id="single index, level=0",
        ),
        pytest.param(
            lambda ix: ix,
            None,
            id="single index, level=None",
        ),
        pytest.param(
            lambda ix: pd.MultiIndex.from_arrays(
                [ix - pd.Timedelta(hours=1), ix], names=["start", "end"]
            ),
            "end",
            id="bar index, level=end",
        ),
        pytest.param(
            lambda ix: pd.MultiIndex.from_arrays(
                [ix, ix + pd.Timedelta(hours=1)], names=["start", "end"]
            ),
            "start",
            id="bar index, level=start",
        ),
        pytest.param(
            lambda ix: pd.MultiIndex.from_arrays(
                [
                    pd.DatetimeIndex(
                        [Timestamp("2022-04-01")] * len(ix),
                        dtype=pd.DatetimeTZDtype(tz="UTC"),
                    ),
                    ix,
                ],
                names=["reference_date", "observation_timestamp"],
            ),
            "observation_timestamp",
            id="bitemporal index, level=observation_timestamp",
        ),
    ],
)
def test_view(
    index_transform: t.Callable[[pd.Index], pd.Index],
    tensor_type: t.Callable[[pd.Series], IndexTensor],
    time_range: TimeRange,
    tensor: pd.Series,
    level: t.Union[int, None],
    expect: pd.Series,
):
    tensor = tensor.copy()
    tensor.index = index_transform(tensor.index)
    tensor = tensor_type(tensor)

    expect = expect.copy()
    expect.index = index_transform(expect.index)
    expect = tensor_type(expect)

    result = time_range.view(tensor, level=level)
    assert_equal_kwargs = (
        dict() if isinstance(result, pd.Index) else dict(check_freq=False)
    )
    assert_equal(result, expect, **assert_equal_kwargs)


base = TimeRange("2000-01-01 00:00", "2000-01-02 00:00")
strict_subset = TimeRange("2000-01-01 10:00", "2000-01-01 12:00")
prefix = TimeRange(base.start, strict_subset.end)
suffix = TimeRange(strict_subset.start, base.end)

disjoint = TimeRange("2000-01-03", "2000-01-04")
superset = TimeRange("1999-12-31", "2000-01-03")
overlaps_start = TimeRange("1999-12-31", "2000-01-01 12:00")
overlaps_end = TimeRange("2000-01-01 12:00", "2000-01-03")


@pytest.mark.parametrize(
    "first, second, expect",
    [
        (base, base, True),
        (base, strict_subset, True),
        (base, prefix, True),
        (base, suffix, True),
        (base, disjoint, False),
        (base, superset, False),
        (base, overlaps_start, False),
        (base, overlaps_end, False),
        (prefix, strict_subset, True),
        (suffix, strict_subset, True),
        (strict_subset, base, False),
        (strict_subset, prefix, False),
        (strict_subset, suffix, False),
        (prefix, suffix, False),
    ],
)
def test_contains(first, second, expect):
    assert first.contains(second) == expect


@pytest.mark.parametrize(
    "first, second, expect",
    [
        (base, base, True),
        (base, strict_subset, True),
        (base, prefix, True),
        (base, suffix, True),
        (base, disjoint, False),
        (base, superset, True),
        (base, overlaps_start, True),
        (base, overlaps_end, True),
    ],
)
def test_intersects(first, second, expect):
    assert first.intersects(second) == expect
    assert second.intersects(first) == expect


@pytest.mark.parametrize(
    "time_range, ts, expect",
    [
        (base, base.start, False),
        (base, base.end, True),
        (base, strict_subset.start, True),
    ],
)
def test_in(time_range, ts, expect):
    assert (ts in time_range) == expect
