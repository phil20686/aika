import typing as t
from abc import ABC, abstractmethod
from functools import reduce

import attr
import pandas as pd
from pandas._libs.tslibs.offsets import BDay, Day, Week, to_offset

from aika.time.time_of_day import TimeOfDay
from aika.time.time_range import RESOLUTION, TimeRange


class ICalendar(ABC):
    """
    Represents a set of timestamps, potentially infinite in size.

    Used to represent the 'expected' index of a dataset.
    """

    @abstractmethod
    def latest_point_before(self, as_of: pd.Timestamp):
        """Return the latest timestamp in `self` which is strictly before `as_of`"""

    @abstractmethod
    def to_index(self, time_range: TimeRange) -> pd.DatetimeIndex:
        """
        Return all the timestamps in `self` within the specified `time_range`, as a
        pandas DatetimeIndex
        """


class Weekdays:
    MON = 0
    TUE = 1
    WED = 2
    THU = 3
    FRI = 4
    SAT = 5
    SUN = 6


BUSINESS_DAYS = (
    Weekdays.MON,
    Weekdays.TUE,
    Weekdays.WED,
    Weekdays.THU,
    Weekdays.FRI,
)


@attr.s(frozen=True)
class TimeOfDayCalendar(ICalendar):

    """
    This expects data at a particular moment in time on a calendar defined by a pandas
    dateOffset class, such as `BDay`, `Week`, `CDay` etc.
    """

    time_of_day: TimeOfDay = attr.ib()
    freq: t.Collection[int] = attr.ib(default=BDay())

    def latest_point_before(self, as_of: pd.Timestamp):

        last_week = self.to_index(
            time_range=TimeRange(
                start=as_of - Week(),
                end=as_of,
            )
        )
        return last_week[-1]

    def _to_date_index(self, time_range):
        start = time_range.start.tz_convert(self.time_of_day.tz).date() - 2 * self.freq
        end = time_range.end.tz_convert(self.time_of_day.tz).date() + 2 * self.freq
        return pd.date_range(start=start, end=end, freq=self.freq)

    def to_index(self, time_range: TimeRange) -> pd.DatetimeIndex:
        dates = self._to_date_index(time_range)
        timestamps = [self.time_of_day.make_timestamp(d) for d in dates]
        return pd.DatetimeIndex([ts for ts in timestamps if ts in time_range])


@attr.s(frozen=True)
class UnionCalendar(ICalendar):

    calendars: t.AbstractSet[ICalendar] = attr.ib(converter=frozenset)

    def latest_point_before(self, as_of: pd.Timestamp):
        return max(calendar.latest_point_before(as_of) for calendar in self.calendars)

    def to_index(self, time_range: TimeRange):
        return reduce(
            pd.Index.union,
            [calendar.to_index(time_range) for calendar in self.calendars],
        )

    @classmethod
    def merge(cls, calendars: t.Iterable[ICalendar]):
        flattened_calendars = set()
        for calendar in calendars:
            if isinstance(calendar, UnionCalendar):
                flattened_calendars.update(calendar.calendars)
            else:
                flattened_calendars.add(calendar)

        return cls(flattened_calendars)


@attr.s(frozen=True)
class OffsetCalendar(ICalendar):

    """
    An offset calendar is used when it is expected that at the time when the graph is run
    there should be at most offset time since the last datapoint.
    """

    offset: pd.offsets.Tick = attr.ib(converter=to_offset)

    # noinspection PyUnresolvedReferences
    @offset.validator
    def _validate_offset(self, attribute, value):
        if not isinstance(value, pd.offsets.Tick):
            raise ValueError(
                f"{attribute} must be a pd.offsets.Tick; got {value} of type "
                f"{value.__class__.__name__}"
            )

        per_day = pd.Timedelta(days=1) / value.delta
        if per_day != int(per_day):
            raise ValueError(
                f"{attribute} must evenly divide into one day; got {value}, of which "
                f"there are a non-integral {per_day} per day"
            )

    def latest_point_before(self, as_of: pd.Timestamp):
        s = pd.Series(dtype="float", index=[as_of - RESOLUTION])
        resampled = s.resample(self.offset).last()
        return resampled.index[-1]

    def to_index(self, time_range: TimeRange):
        start = self.latest_point_before(time_range.start) + self.offset
        return pd.date_range(
            start=start,
            end=time_range.end - RESOLUTION,
            freq=self.offset,
        )
