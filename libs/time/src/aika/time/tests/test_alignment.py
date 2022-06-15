import numpy as np
import pandas as pd
import pytest

from aika.utilities.testing import assert_call

from aika.time.alignment import causal_match, causal_resample
from aika.time.timestamp import Timestamp


class PdIndex:
    date_index = pd.date_range("2000-01-01", periods=10)
    hourly_index = pd.date_range("2000-01-01", periods=10, freq="1H")
    duplicated_hourly_index = hourly_index.append(hourly_index).sort_values()
    half_hourly_index = pd.date_range("2000-01-01", periods=10, freq="30Min")
    half_hourly_index_offset = pd.date_range(
        "2000-01-01T00:00:05", periods=10, freq="30Min"
    )


class AikaIndex:
    date_index = pd.DatetimeIndex([Timestamp(x) for x in PdIndex.date_index])
    hourly_index = pd.DatetimeIndex([Timestamp(x) for x in PdIndex.hourly_index])
    duplicated_hourly_index = pd.DatetimeIndex(
        [Timestamp(x) for x in PdIndex.duplicated_hourly_index]
    )
    half_hourly_index = pd.DatetimeIndex(
        [Timestamp(x) for x in PdIndex.half_hourly_index]
    )
    half_hourly_index_offset = pd.DatetimeIndex(
        [Timestamp(x) for x in PdIndex.half_hourly_index_offset]
    )


@pytest.mark.parametrize(
    "data, index, contemp, fill_limit, expected",
    [
        (
            pd.Series(
                range(10),
                index=PdIndex.date_index,
            ),
            PdIndex.date_index,
            True,
            None,
            pd.Series(
                range(10),
                index=PdIndex.date_index,
            ),
        ),
        (
            pd.Series(
                range(10),
                index=PdIndex.date_index,
            ),
            PdIndex.date_index,
            False,
            None,
            pd.Series(
                [np.nan] + list(range(9)),
                index=PdIndex.date_index,
            ),
        ),
        (
            pd.Series(
                range(10),
                index=PdIndex.hourly_index,
            ),
            PdIndex.duplicated_hourly_index,
            True,
            0,
            pd.concat(
                [
                    pd.Series(
                        range(10),
                        index=PdIndex.hourly_index,
                    ),
                    pd.Series(
                        range(10),
                        index=PdIndex.hourly_index,
                    ),
                ],
                axis=0,
            ).sort_index(),
        ),
        (
            pd.Series(range(10), index=PdIndex.hourly_index),
            PdIndex.date_index,
            True,
            0,
            pd.Series([0, 9] + [np.nan] * 8, index=PdIndex.date_index),
        ),
        (
            pd.Series(range(10), index=PdIndex.hourly_index),
            PdIndex.date_index,
            True,
            3,
            pd.Series([0, 9, 9, 9, 9] + [np.nan] * 5, index=PdIndex.date_index),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=PdIndex.half_hourly_index),
                    "B": pd.Series(2.0, index=PdIndex.half_hourly_index_offset),
                }
            ),
            PdIndex.half_hourly_index,
            True,
            None,
            pd.DataFrame(
                [[1.0, np.nan]] * 10,
                index=PdIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=PdIndex.half_hourly_index),
                    "B": pd.Series(2.0, index=PdIndex.half_hourly_index_offset),
                }
            ),
            PdIndex.half_hourly_index,
            False,
            None,
            pd.DataFrame(
                [[np.nan, np.nan]] + [[np.nan, 2.0]] * 9,
                index=PdIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=PdIndex.half_hourly_index[:5]),
                    "B": pd.Series(2.0, index=PdIndex.half_hourly_index_offset[:5]),
                }
            ),
            PdIndex.half_hourly_index,
            False,
            0,
            pd.DataFrame(
                [[np.nan, np.nan]] + [[np.nan, 2.0]] * 5 + [[np.nan, np.nan]] * 4,
                index=PdIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        # Now duplicate with aika time stamps:
        (
            pd.Series(
                range(10),
                index=AikaIndex.date_index,
            ),
            AikaIndex.date_index,
            True,
            None,
            pd.Series(
                range(10),
                index=AikaIndex.date_index,
            ),
        ),
        (
            pd.Series(
                range(10),
                index=AikaIndex.date_index,
            ),
            AikaIndex.date_index,
            False,
            None,
            pd.Series(
                [np.nan] + list(range(9)),
                index=AikaIndex.date_index,
            ),
        ),
        (
            pd.Series(
                range(10),
                index=AikaIndex.hourly_index,
            ),
            AikaIndex.duplicated_hourly_index,
            True,
            0,
            pd.concat(
                [
                    pd.Series(
                        range(10),
                        index=AikaIndex.hourly_index,
                    ),
                    pd.Series(
                        range(10),
                        index=AikaIndex.hourly_index,
                    ),
                ],
                axis=0,
            ).sort_index(),
        ),
        (
            pd.Series(range(10), index=AikaIndex.hourly_index),
            AikaIndex.date_index,
            True,
            0,
            pd.Series([0, 9] + [np.nan] * 8, index=AikaIndex.date_index),
        ),
        (
            pd.Series(range(10), index=AikaIndex.hourly_index),
            AikaIndex.date_index,
            True,
            3,
            pd.Series([0, 9, 9, 9, 9] + [np.nan] * 5, index=AikaIndex.date_index),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=AikaIndex.half_hourly_index),
                    "B": pd.Series(2.0, index=AikaIndex.half_hourly_index_offset),
                }
            ),
            AikaIndex.half_hourly_index,
            True,
            None,
            pd.DataFrame(
                [[1.0, np.nan]] * 10,
                index=AikaIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=AikaIndex.half_hourly_index),
                    "B": pd.Series(2.0, index=AikaIndex.half_hourly_index_offset),
                }
            ),
            AikaIndex.half_hourly_index,
            False,
            None,
            pd.DataFrame(
                [[np.nan, np.nan]] + [[np.nan, 2.0]] * 9,
                index=AikaIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=AikaIndex.half_hourly_index[:5]),
                    "B": pd.Series(2.0, index=AikaIndex.half_hourly_index_offset[:5]),
                }
            ),
            AikaIndex.half_hourly_index,
            False,
            0,
            pd.DataFrame(
                [[np.nan, np.nan]] + [[np.nan, 2.0]] * 5 + [[np.nan, np.nan]] * 4,
                index=AikaIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        # assert failure when you mix time zones.
        (
            pd.Series(
                range(10),
                index=PdIndex.date_index,
            ),
            AikaIndex.date_index,
            True,
            None,
            ValueError,
        ),
    ],
)
def test_causal_match_single_index(data, index, contemp, fill_limit, expected):
    assert_call(
        causal_match,
        expected,
        data=data,
        index=index,
        contemp=contemp,
        fill_limit=fill_limit,
    )
    assert_call(
        causal_match,
        expected,
        data=data,
        index=pd.Series(1.0, index=index),
        contemp=contemp,
        fill_limit=fill_limit,
    )
    assert_call(
        causal_match,
        expected,
        data=data,
        index=pd.DataFrame(1.0, index=index, columns=list("AB")),
        contemp=contemp,
        fill_limit=fill_limit,
    )


@pytest.mark.parametrize("agg_method", ["last"])
@pytest.mark.parametrize(
    "data, index, contemp, expected",
    [
        (
            pd.Series(
                range(10),
                index=PdIndex.date_index,
            ),
            PdIndex.date_index,
            True,
            pd.Series(
                range(10),
                index=PdIndex.date_index,
            ),
        ),
        (
            pd.Series(
                range(10),
                index=PdIndex.date_index,
            ),
            PdIndex.date_index,
            False,
            pd.Series(
                [np.nan] + list(range(9)),
                index=PdIndex.date_index,
            ),
        ),
        (
            pd.Series(range(10), index=PdIndex.hourly_index),
            PdIndex.date_index,
            True,
            pd.Series([0, 9] + [np.nan] * 8, index=PdIndex.date_index),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=PdIndex.half_hourly_index),
                    "B": pd.Series(2.0, index=PdIndex.half_hourly_index_offset),
                }
            ),
            PdIndex.half_hourly_index,
            True,
            pd.DataFrame(
                [[1.0, np.nan]] + [[1.0, 2.0]] * 9,
                index=PdIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=PdIndex.half_hourly_index),
                    "B": pd.Series(2.0, index=PdIndex.half_hourly_index_offset),
                }
            ),
            PdIndex.half_hourly_index,
            False,
            pd.DataFrame(
                [[np.nan, np.nan]] + [[1.0, 2.0]] * 9,
                index=PdIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=PdIndex.half_hourly_index[:5]),
                    "B": pd.Series(2.0, index=PdIndex.half_hourly_index_offset[:5]),
                }
            ),
            PdIndex.half_hourly_index,
            False,
            pd.DataFrame(
                [[np.nan, np.nan]] + [[1.0, 2.0]] * 5 + [[np.nan, np.nan]] * 4,
                index=PdIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.Series(
                range(10),
                index=PdIndex.hourly_index,
            ),
            PdIndex.duplicated_hourly_index,
            True,
            pd.concat(
                [
                    pd.Series(
                        range(10),
                        index=PdIndex.hourly_index,
                    ),
                    pd.Series(
                        range(10),
                        index=PdIndex.hourly_index,
                    ),
                ],
                axis=0,
            ).sort_index(),
        ),
        # Now duplicate with aika time stamps:
        (
            pd.Series(
                range(10),
                index=AikaIndex.date_index,
            ),
            AikaIndex.date_index,
            True,
            pd.Series(
                range(10),
                index=AikaIndex.date_index,
            ),
        ),
        (
            pd.Series(
                range(10),
                index=AikaIndex.date_index,
            ),
            AikaIndex.date_index,
            False,
            pd.Series(
                [np.nan] + list(range(9)),
                index=AikaIndex.date_index,
            ),
        ),
        (
            pd.Series(range(10), index=AikaIndex.hourly_index),
            AikaIndex.date_index,
            True,
            pd.Series([0, 9] + [np.nan] * 8, index=AikaIndex.date_index),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=AikaIndex.half_hourly_index),
                    "B": pd.Series(2.0, index=AikaIndex.half_hourly_index_offset),
                }
            ),
            AikaIndex.half_hourly_index,
            True,
            pd.DataFrame(
                [[1.0, np.nan]] + [[1.0, 2.0]] * 9,
                index=AikaIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=AikaIndex.half_hourly_index),
                    "B": pd.Series(2.0, index=AikaIndex.half_hourly_index_offset),
                }
            ),
            AikaIndex.half_hourly_index,
            False,
            pd.DataFrame(
                [[np.nan, np.nan]] + [[1.0, 2.0]] * 9,
                index=AikaIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(1.0, index=AikaIndex.half_hourly_index[:5]),
                    "B": pd.Series(2.0, index=AikaIndex.half_hourly_index_offset[:5]),
                }
            ),
            AikaIndex.half_hourly_index,
            False,
            pd.DataFrame(
                [[np.nan, np.nan]] + [[1.0, 2.0]] * 5 + [[np.nan, np.nan]] * 4,
                index=AikaIndex.half_hourly_index,
                columns=list("AB"),
            ),
        ),
        (
            pd.Series(
                range(10),
                index=AikaIndex.hourly_index,
            ),
            AikaIndex.duplicated_hourly_index,
            True,
            pd.concat(
                [
                    pd.Series(
                        range(10),
                        index=AikaIndex.hourly_index,
                    ),
                    pd.Series(
                        range(10),
                        index=AikaIndex.hourly_index,
                    ),
                ],
                axis=0,
            ).sort_index(),
        ),
        # assert failure when you mix time zones.
        (
            pd.Series(
                range(10),
                index=PdIndex.date_index,
            ),
            AikaIndex.date_index,
            True,
            ValueError,
        ),
    ],
)
def test_causal_resample_single_index(data, index, contemp, agg_method, expected):
    assert_call(
        causal_resample,
        expected,
        data=data,
        agg_method=agg_method,
        index=index,
        contemp=contemp,
    )
    assert_call(
        causal_resample,
        expected,
        data=data,
        agg_method=agg_method,
        index=pd.Series(1.0, index=index),
        contemp=contemp,
    )
    assert_call(
        causal_resample,
        expected,
        data=data,
        agg_method=agg_method,
        index=pd.DataFrame(1.0, index=index, columns=list("AB")),
        contemp=contemp,
    )


@pytest.mark.parametrize(
    "data, data_level, index, index_level, contemp, fill_limit, expect",
    [
        (
            pd.Series(range(10), index=PdIndex.date_index),
            None,
            pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            1,
            True,
            0,
            pd.Series(
                data=list(range(10)) * 3,
                index=pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            ),
        ),
        (
            pd.Series(range(10), index=PdIndex.date_index),
            None,
            pd.MultiIndex.from_product([PdIndex.date_index, list("ABC")]),
            0,
            True,
            0,
            pd.Series(
                data=list(range(10)) * 3,
                index=pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            )
            .swaplevel(0, 1)
            .sort_index(),
        ),
        (
            pd.Series(range(10), index=PdIndex.date_index),
            None,
            pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            1,
            False,
            0,
            pd.Series(
                data=([np.nan] + list(range(9))) * 3,
                index=pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            ),
        ),
    ],
)
def test_causal_match_multiindex(
    data, data_level, index, index_level, contemp, fill_limit, expect
):
    assert_call(
        causal_match,
        expect,
        data=data,
        data_level=data_level,
        index=index,
        index_level=index_level,
        contemp=contemp,
        fill_limit=fill_limit,
    )
    assert_call(
        causal_match,
        expect,
        data=data,
        data_level=data_level,
        index=pd.Series(1.0, index=index),
        index_level=index_level,
        contemp=contemp,
        fill_limit=fill_limit,
    )
    assert_call(
        causal_match,
        expect,
        data=data,
        data_level=data_level,
        index=pd.DataFrame(1.0, index=index, columns=list("AB")),
        index_level=index_level,
        contemp=contemp,
        fill_limit=fill_limit,
    )


@pytest.mark.parametrize("agg_method", ["last"])
@pytest.mark.parametrize(
    "data, data_level, index, index_level, contemp, expect",
    [
        (
            pd.Series(range(10), index=PdIndex.date_index),
            None,
            pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            1,
            True,
            pd.Series(
                data=list(range(10)) * 3,
                index=pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            ),
        ),
        (
            pd.Series(range(10), index=PdIndex.date_index),
            None,
            pd.MultiIndex.from_product([PdIndex.date_index, list("ABC")]),
            0,
            True,
            pd.Series(
                data=list(range(10)) * 3,
                index=pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            )
            .swaplevel(0, 1)
            .sort_index(),
        ),
        (
            pd.Series(range(10), index=PdIndex.date_index),
            None,
            pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            1,
            False,
            pd.Series(
                data=([np.nan] + list(range(9))) * 3,
                index=pd.MultiIndex.from_product([list("ABC"), PdIndex.date_index]),
            ),
        ),
    ],
)
def test_causal_match_multiindex(
    data,
    data_level,
    index,
    index_level,
    contemp,
    expect,
    agg_method,
):
    assert_call(
        causal_resample,
        expect,
        agg_method=agg_method,
        data=data,
        data_level=data_level,
        index=index,
        index_level=index_level,
        contemp=contemp,
    )
    assert_call(
        causal_resample,
        expect,
        agg_method=agg_method,
        data=data,
        data_level=data_level,
        index=pd.Series(1.0, index=index),
        index_level=index_level,
        contemp=contemp,
    )
    assert_call(
        causal_resample,
        expect,
        data=data,
        agg_method=agg_method,
        data_level=data_level,
        index=pd.DataFrame(1.0, index=index, columns=list("AB")),
        index_level=index_level,
        contemp=contemp,
    )
