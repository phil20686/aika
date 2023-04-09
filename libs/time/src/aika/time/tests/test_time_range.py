import typing as t

import pandas as pd
import pytest

from aika.time.time_range import RESOLUTION, TimeRange
from aika.time.timestamp import Timestamp
from aika.utilities.pandas_utils import IndexTensor, Tensor
from aika.utilities.testing import assert_call, assert_equal
from packaging import version


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
        )
        if version.parse(pd.__version__) >= version.parse("1.2.0")
        else (
            None,
            "2020-01-01T12:00 [America/New_York]",
            TimeRange(None, "2020-01-01T12:00 [America/New_York]"),
            "TimeRange('1677-09-21T00:12:43.145225 [UTC]', '2020-01-01T12:00:00 [America/New_York]')",
        ),  # unclear why this is necessary? seems python 3.6 restricts us to very old pandas/numpy versions,
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
        assert val == TimeRange.from_string(str_repr)


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


@pytest.mark.parametrize(
    "tr, tensor, level, expect",
    [
        (
            TimeRange("2000-01-01 00:00", "2000-01-04 00:00"),
            pd.Series(
                1.0,
                index=pd.MultiIndex.from_product(
                    [
                        list("ABC"),
                        pd.date_range(
                            start="1999-12-20 12:00",
                            end="2000-01-05",
                            freq="D",
                            tz="UTC",
                        ),
                    ],
                    names=("foo", "time_index"),
                ),
            ),
            "time_index",
            pd.Series(
                1.0,
                index=pd.MultiIndex.from_product(
                    [
                        list("ABC"),
                        pd.date_range(
                            start="2000-01-01 12:00",
                            end="2000-01-04 00:00",
                            freq="D",
                            tz="UTC",
                        ),
                    ],
                    names=("foo", "time_index"),
                ),
            ),
        ),
        (
            TimeRange("2000-01-01 00:00", "2000-01-04 00:00"),
            pd.Series(
                1.0,
                index=pd.MultiIndex.from_product(
                    [
                        list("ABC"),
                        pd.date_range(
                            start="1999-12-20 12:00",
                            end="2000-01-05",
                            freq="D",
                            tz="UTC",
                        ),
                    ],
                    names=("foo", "time_index"),
                ),
            ),
            None,
            ValueError("Must specify `level` if tensor is multi-indexed."),
        ),
    ],
)
def test_view_error(tr, tensor, level, expect):
    assert_call(tr.view, expect, tensor, level)


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
    assert_call(first.intersects, expect, second)
    assert_call(second.intersects, expect, first)


@pytest.mark.parametrize(
    "time_range, ts, expect",
    [
        (base, base.start, True),
        (base, base.end, False),
        (base, strict_subset.start, True),
    ],
)
def test_in(time_range, ts, expect):
    assert (ts in time_range) == expect


@pytest.mark.parametrize(
    "index_tensor, level, expect",
    [
        (
            pd.DataFrame(),
            None,
            ValueError("Cannot extract time range from empty index"),
        ),
        (
            timestamp_index(start="2012-01-01 00:00", freq="H", periods=10),
            None,
            TimeRange("2012-01-01 00:00", "2012-01-01 09:00:00.000000001"),
        ),
    ],
)
def test_from_pandas(index_tensor: IndexTensor, level, expect):
    assert_call(TimeRange.from_pandas, expect, index_tensor, level)


@pytest.mark.parametrize(
    "first, second, expect",
    [
        (base, strict_subset, base),
        (base, disjoint, ValueError("Cannot union non-intersecting time ranges")),
        (base, superset, superset),
        (base, overlaps_start, TimeRange(overlaps_start.start, base.end)),
        (base, overlaps_end, TimeRange(base.start, overlaps_end.end)),
    ],
)
def test_union(first, second, expect):
    assert_call(
        first.union,
        expect,
        second,
    )
    # because union is symmetric
    assert_call(second.union, expect, first)


@pytest.mark.parametrize(
    "first, second, expect",
    [
        (base, strict_subset, strict_subset),
        (
            base,
            disjoint,
            ValueError("Cannot give intersection of non-intersecting time-ranges"),
        ),
        (base, superset, base),
        (base, overlaps_start, TimeRange(base.start, overlaps_start.end)),
        (base, overlaps_end, TimeRange(overlaps_end.start, base.end)),
    ],
)
def test_intersection(first, second, expect):
    assert_call(
        first.intersection,
        expect,
        second,
    )
    # because union is symmetric
    assert_call(second.intersection, expect, first)
