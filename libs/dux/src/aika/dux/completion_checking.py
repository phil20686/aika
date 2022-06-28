import typing as t

import attr

from aika.datagraph.interface import DataSetMetadata
from aika.time.calendars import ICalendar, UnionCalendar
from aika.time.time_range import TimeRange

from aika.dux.interface import Dependency, ICompletionChecker, ITimeSeriesTask


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


def infer_inherited_completion_checker(dependencies: t.Mapping[str, Dependency]):

    time_series_dependencies: t.Dict[str, Dependency[ITimeSeriesTask]] = {
        name: dep for name, dep in dependencies.items() if dep.task.time_series
    }

    explicit_inheritors = set()
    explicit_non_inheritors = set()

    for name, dep in time_series_dependencies.items():
        if dep.inherit_frequency:
            explicit_inheritors.add(name)
        elif dep.inherit_frequency is not None:
            explicit_non_inheritors.add(name)

    inheritors = (
        explicit_inheritors
        if explicit_inheritors
        else (set(time_series_dependencies) - explicit_non_inheritors)
    )

    completion_checkers: t.Dict[str, ICompletionChecker] = {
        name: time_series_dependencies[name].task.completion_checker
        for name in inheritors
    }

    if len(completion_checkers) == 0:
        raise ValueError(
            f"Task has no dependencies to inherit its completion_checker from; "
            f"completion_checker must be specified explicitly for this task."
        )

    elif len(completion_checkers) == 1:
        (result,) = completion_checkers.values()
        return result

    elif all(isinstance(cc, CalendarChecker) for cc in completion_checkers):
        completion_checkers: t.Dict[str, CalendarChecker]
        calendar = UnionCalendar.merge(
            [cc.calendar for cc in completion_checkers.values()]
        )
        return CalendarChecker(calendar)

    elif all(isinstance(cc, IrregularChecker) for cc in completion_checkers):
        # note that this branch is technically redundant since IrregularChecker()
        # is a singleton and hence this case will always be covered by the len == 1
        # branch.
        return IrregularChecker()

    else:
        regular = {
            name
            for name, checker in completion_checkers.items()
            if isinstance(checker, CalendarChecker)
        }

        irregular = set(completion_checkers) - regular

        raise ValueError(
            f"Task has inconsistent completion checkers among its dependencies; "
            f"dependencies {regular} all have {CalendarChecker.__name__} completion "
            f"checkers, while dependencies {irregular} have "
            f"{IrregularChecker.__name__} completion checkers. These cannot be "
            f"generically combined, so an inherited completion checker cannot be "
            f"inferred.\n To fix this, either specify `completion_checker` explicitly "
            f"for this task, or update the `inherit_frequency` settings on the "
            f"dependencies to ensure that the dependencies being inherited from have a "
            "are either all regular or all irregular."
        )
