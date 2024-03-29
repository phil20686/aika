import attr

from aika.datagraph.interface import DataSetMetadata
from aika.putki.interface import Dependency, ICompletionChecker, ITimeSeriesTask
from aika.time.calendars import ICalendar, UnionCalendar
from aika.time.time_range import TimeRange


@attr.s(frozen=True)
class CalendarChecker(ICompletionChecker):
    """
    Used for checking against a calendar. A calendar is an object that, given a target time range,
    can yield the last expected index point. See aika.time.calendars for more information.
    """

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
    """
    Checker that only cares about the target time range that a task has been run for
    - it ignores the actual data. Useful for data with no known schedule.
    """

    def is_complete(
        self,
        metadata: "DataSetMetadata",
        target_time_range: TimeRange,
    ) -> bool:
        if not metadata.exists():
            return False

        declared_time_range = metadata.get_declared_time_range()
        if not declared_time_range.intersects(target_time_range):
            raise ValueError(
                "Increments should not be run with non-overlapping time ranges."
            )
        return declared_time_range.end >= target_time_range.end
