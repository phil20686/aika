import pandas as pd
import pytest

from aika.utilities.testing import assert_call

from aika.time.timestamp import Timestamp


@pytest.mark.parametrize(
    "ts, expect",
    [
        (  # dates become timestamps
            "2000-01-01",
            pd.Timestamp(year=2000, month=1, day=1, hour=0, tz="UTC"),
        ),
        (  # all timestamps assumed UTC if not specified in the string
            "2000-01-01 12:00",
            pd.Timestamp(year=2000, month=1, day=1, hour=12, tz="UTC"),
        ),
        (  # if specified in the string the specifiged time stamps are used
            "2012-01-01 16:00 [Europe/London]",
            pd.Timestamp(year=2012, month=1, day=1, hour=16, tz="Europe/London"),
        ),
        (  # If a naive timestamp is passed in it is converted to UTC
            pd.Timestamp(year=2016, month=6, day=3, hour=12, tz=None),
            pd.Timestamp(year=2016, month=6, day=3, hour=12, tz="UTC"),
        ),
        (  # If a timezone with a timestamp is passed in it passed through unaltered.
            pd.Timestamp(year=2016, month=6, day=3, hour=12, tz="America/New_York"),
            pd.Timestamp(year=2016, month=6, day=3, hour=12, tz="America/New_York"),
        ),
        (
            # Test microsecond second resolution
            "2020-01-01T12:00:00.000001",
            pd.Timestamp(year=2020, month=1, day=1, hour=12, microsecond=1, tz="UTC"),
        ),
        (
            # Test nanosecond second resolution
            "2020-01-01T12:00:00.000000001",
            pd.Timestamp(year=2020, month=1, day=1, hour=12, nanosecond=1, tz="UTC"),
        ),
    ],
)
def test_timestamp_constructor(ts, expect):
    print(Timestamp(ts))
    print(expect)
    assert_call(Timestamp, expect, ts)
