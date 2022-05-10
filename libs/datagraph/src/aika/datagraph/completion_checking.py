from abc import ABC

import attr

from aika.time.calendars import ICalendar
from aika.time.time_range import TimeRange

from aika.datagraph.interface import DataSetMetadata


class ICompletionChecker(ABC):
    def is_complete(
        self,
        metadata: "DataSetMetadata",
        target_time_range: TimeRange,
    ) -> bool:
        pass


@attr.s(frozen=True)
class CalendarChecker(ICompletionChecker):

    calendar: ICalendar = attr.ib()

    def is_complete(
        self,
        metadata: "DataSetMetadata",
        target_time_range: TimeRange,
    ) -> bool:
        if not metadata.exists():
            return False

        required_time_range = TimeRange(
            start=self.calendar.latest_point_before(target_time_range.end),
            end=target_time_range.end,
        )
        actual_time_range: TimeRange = metadata.get_data_time_range()
        return required_time_range.intersects(actual_time_range)


@attr.s(frozen=True)
class IrregularChecker(ICompletionChecker):
    def is_complete(
        self,
        metadata: "DataSetMetadata",
        target_time_range: TimeRange,
    ) -> bool:
        if not metadata.exists():
            return False

        declared_time_range = metadata.get_declared_time_range()
        return declared_time_range.end >= target_time_range.end
