import pandas as pd
import pytest
from pandas.tseries.offsets import Hour, Minute, Week

from aika.time.calendars import (
    ICalendar,
    OffsetCalendar,
    TimeOfDayCalendar,
    UnionCalendar,
    Weekdays,
)
from aika.time.time_of_day import TimeOfDay
from aika.time.time_range import RESOLUTION, TimeRange
from aika.time.timestamp import Timestamp


@pytest.mark.parametrize(
    "calendar",
    [
        TimeOfDayCalendar(TimeOfDay.from_str("15:45")),
        TimeOfDayCalendar(TimeOfDay.from_str("15:45 [Europe/London]")),
        TimeOfDayCalendar(TimeOfDay.from_str("15:45 [America/New_York]")),
        TimeOfDayCalendar(
            TimeOfDay.from_str("15:45 [Europe/London]"),
            weekdays=(Weekdays.MON, Weekdays.THU),
        ),
        OffsetCalendar("5MIN"),
        OffsetCalendar("1MIN"),
        OffsetCalendar(Hour()),
        UnionCalendar(
            {
                TimeOfDayCalendar(TimeOfDay.from_str("15:47 [Europe/London]")),
                TimeOfDayCalendar(TimeOfDay.from_str("15:47 [America/New_York]")),
                OffsetCalendar("5MIN"),
            }
        ),
    ],
)
@pytest.mark.parametrize(
    "timestamp",
    [
        Timestamp("2022-04-28 15:45"),
        Timestamp("2022-04-28 15:46"),
        Timestamp("2022-04-29 15:44"),
        Timestamp("2022-04-29 15:45"),
        Timestamp("2022-04-29 15:50"),
        Timestamp("2022-04-30 15:00"),
        Timestamp("2022-05-01 15:00"),
        Timestamp("2022-05-02 15:00"),
        Timestamp("2022-05-02 15:50"),
        Timestamp("2022-05-02 15:48"),
        Timestamp("2022-05-02 15:51"),
    ],
)
def test_consistency(calendar: ICalendar, timestamp: pd.Timestamp):
    latest_point_before = calendar.latest_point_before(timestamp)
    index = calendar.to_index(TimeRange("2022-01-01", timestamp))
    assert latest_point_before == index[-1]


_timezones = [
    "UTC",
    "Europe/London",
    "America/New_York",
    "Pacific/Pago_Pago",
]


@pytest.mark.parametrize(
    "calendar, timestamp, expect",
    [
        *[
            (TimeOfDayCalendar(TimeOfDay.from_str(f"15:45 [{tz}]")), timestamp, expect)
            for tz in _timezones
            for timestamp, expect in [
                (
                    Timestamp(f"2022-04-29 15:40 [{tz}]"),
                    Timestamp(f"2022-04-28 15:45 [{tz}]"),
                ),
                (
                    Timestamp(f"2022-04-29 15:45 [{tz}]"),
                    Timestamp(f"2022-04-28 15:45 [{tz}]"),
                ),
                (
                    Timestamp(f"2022-04-29 15:45 [{tz}]") + RESOLUTION,
                    Timestamp(f"2022-04-29 15:45 [{tz}]"),
                ),
                (
                    Timestamp(f"2022-04-29 15:46 [{tz}]"),
                    Timestamp(f"2022-04-29 15:45 [{tz}]"),
                ),
                # weekends are skipped with default weekday mask
                (
                    Timestamp(f"2022-04-30 15:46 [{tz}]"),  # Saturday
                    Timestamp(f"2022-04-29 15:45 [{tz}]"),
                ),
                (
                    Timestamp(f"2022-05-01 15:46 [{tz}]"),  # Sunday
                    Timestamp(f"2022-04-29 15:45 [{tz}]"),
                ),
                (
                    Timestamp(f"2022-05-02 15:46 [{tz}]"),  # Monday
                    Timestamp(f"2022-05-02 15:45 [{tz}]"),
                ),
            ]
        ],
        *[
            (
                TimeOfDayCalendar(
                    TimeOfDay.from_str(f"15:45 [{tz}]"),
                    weekdays=(Weekdays.FRI, Weekdays.SAT),
                ),
                timestamp,
                expect,
            )
            for tz in _timezones
            for timestamp, expect in [
                #
                (
                    pd.Timestamp("2022-04-28 15:46", tz=tz),  # Thursday
                    pd.Timestamp("2022-04-23 15:45", tz=tz),  # Saturday
                ),
                (
                    pd.Timestamp("2022-04-29 15:44", tz=tz),  # Friday, before time
                    pd.Timestamp("2022-04-23 15:45", tz=tz),  # Saturday
                ),
                (
                    pd.Timestamp("2022-04-29 15:46", tz=tz),  # Friday, after time
                    pd.Timestamp("2022-04-29 15:45", tz=tz),  # Friday
                ),
                (
                    pd.Timestamp("2022-04-30 15:44", tz=tz),  # Saturday, before time
                    pd.Timestamp("2022-04-29 15:45", tz=tz),  # Friday
                ),
                (
                    pd.Timestamp(
                        "2022-04-30 15:46", tz=tz
                    ),  # Saturday, after specified time
                    pd.Timestamp("2022-04-30 15:45", tz=tz),  # Saturday
                ),
                (
                    pd.Timestamp("2022-05-01 15:46", tz=tz),  # Sunday
                    pd.Timestamp("2022-04-30 15:45", tz=tz),  # Saturday
                ),
                (
                    pd.Timestamp("2022-05-02 15:46", tz=tz),  # Monday
                    pd.Timestamp("2022-04-30 15:45", tz=tz),  # Saturday
                ),
            ]
        ],
        *[
            (
                UnionCalendar.merge(
                    [
                        TimeOfDayCalendar(
                            TimeOfDay.from_str("15:45 [Europe/London]"),
                            weekdays=[Weekdays.FRI],
                        ),
                        UnionCalendar(
                            {
                                TimeOfDayCalendar(TimeOfDay.from_str("09:20 [UTC]")),
                                TimeOfDayCalendar(TimeOfDay.from_str("09:30 [UTC]")),
                            }
                        ),
                    ]
                ),
                timestamp,
                expect,
            )
            for timestamp, expect in [
                (
                    Timestamp("2022-04-29 09:00 [UTC]"),  # Friday
                    Timestamp("2022-04-28 09:30 [UTC]"),  # Thursday
                ),
                (
                    Timestamp("2022-04-29 09:20 [UTC]"),
                    Timestamp("2022-04-28 09:30 [UTC]"),
                ),
                (
                    Timestamp("2022-04-29 09:20 [UTC]") + RESOLUTION,
                    Timestamp("2022-04-29 09:20 [UTC]"),
                ),
                (
                    Timestamp("2022-04-29 09:29 [UTC]"),
                    Timestamp("2022-04-29 09:20 [UTC]"),
                ),
                (
                    Timestamp("2022-04-29 09:31 [UTC]"),
                    Timestamp("2022-04-29 09:30 [UTC]"),
                ),
                (
                    Timestamp("2022-04-29 15:40 [Europe/London]"),
                    Timestamp("2022-04-29 09:30 [UTC]"),
                ),
                (
                    Timestamp("2022-04-29 15:40 [Europe/London]"),
                    Timestamp("2022-04-29 09:30 [UTC]"),
                ),
                (
                    Timestamp("2022-04-29 15:45:01 [Europe/London]"),
                    Timestamp("2022-04-29 15:45 [Europe/London]"),
                ),
                (
                    Timestamp("2022-04-30 09:00 [UTC]"),  # Saturday
                    Timestamp("2022-04-29 15:45 [Europe/London]"),
                ),
                (
                    Timestamp("2022-05-01 09:00 [UTC]"),  # Sunday
                    Timestamp("2022-04-29 15:45 [Europe/London]"),
                ),
                (
                    Timestamp("2022-05-02 09:00 [UTC]"),  # Monday
                    Timestamp("2022-04-29 15:45 [Europe/London]"),
                ),
                (
                    Timestamp("2022-05-02 10:00 [UTC]"),  # Monday
                    Timestamp("2022-05-02 09:30 [UTC]"),
                ),
                (
                    Timestamp("2022-05-03 09:00 [UTC]"),  # Tuesday
                    Timestamp("2022-05-02 09:30 [UTC]"),
                ),
            ]
        ],
        *[
            (
                OffsetCalendar(Minute(5)),
                Timestamp("2022-04-29 12:03"),
                Timestamp("2022-04-29 12:00"),
            ),
            (
                OffsetCalendar(Minute(1)),
                Timestamp("2022-04-29 12:03"),
                Timestamp("2022-04-29 12:02"),
            ),
            (
                OffsetCalendar(Minute(1)),
                Timestamp("2022-04-29 12:03") + RESOLUTION,
                Timestamp("2022-04-29 12:03"),
            ),
            (
                OffsetCalendar(Hour()),
                Timestamp("2022-04-29 12:43"),
                Timestamp("2022-04-29 12:00"),
            ),
            (
                OffsetCalendar(Hour()),
                Timestamp("2022-04-29 12:00"),
                Timestamp("2022-04-29 11:00"),
            ),
        ],
    ],
)
@pytest.mark.parametrize(
    "tz",
    _timezones,
)
def test_latest_point_before(calendar, timestamp, tz, expect):
    localized_timestamp = timestamp.tz_convert(tz)
    result = calendar.latest_point_before(localized_timestamp)
    assert result == expect
    assert calendar.to_index(TimeRange(timestamp - Week(), timestamp))[-1] == result
