import datetime

import pytest as pytest
import pytz

from aika.time.time_of_day import TimeOfDay
from aika.time.timestamp import Timestamp
from aika.utilities.testing import assert_call


@pytest.mark.parametrize(
    "input_string, expect",
    [
        (
            "21:00:05.5 [America/New_York]",
            TimeOfDay(
                datetime.time(21, 0, 5, microsecond=500000),
                tz=pytz.timezone("America/New_York"),
            ),
        ),
        (
            "21:00:05.5 [America/New_York]",
            TimeOfDay(
                datetime.time(21, 0, 5, microsecond=500000), tz="America/New_York"
            ),
        ),
        ("21,00:05.5 America/New_York]", ValueError("Failed to parse.*")),
        ("21,00:05.5 [America/New_York]", ValueError("Time string.*did not match.*")),
    ],
)
def test_time_of_day_builder(input_string, expect):
    assert_call(TimeOfDay.from_str, expect, input_string)


@pytest.mark.parametrize(
    "time_of_day, date, expect",
    [
        (
            TimeOfDay.from_str("21:01:02.345 [America/New_York]"),
            datetime.date(2012, 1, 5),
            Timestamp("2012-01-05T21:01:02.345 [America/New_York]"),
        )
    ],
)
def test_make_timestamp(time_of_day, date, expect):
    assert_call(time_of_day.make_timestamp, expect, date)
