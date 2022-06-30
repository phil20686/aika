from unittest.mock import Mock

import pytest

from aika.time.calendars import TimeOfDayCalendar, UnionCalendar
from aika.time.time_of_day import TimeOfDay
from aika.time.time_range import TimeRange
from aika.utilities.testing import assert_call

from aika.putki import CalendarChecker, ICompletionChecker, IrregularChecker
from aika.putki.completion_checking import infer_inherited_completion_checker


def _mock_metadata(data_time_range, declared_time_range=None, exists=True):
    mock = Mock()
    mock.exists = Mock(return_value=exists)
    mock.get_data_time_range = Mock(return_value=data_time_range)
    mock.get_declared_time_range = Mock(return_value=declared_time_range)
    return mock


def _time_of_day_checker(time_of_day: str):
    return CalendarChecker(TimeOfDayCalendar(TimeOfDay.from_str(time_of_day)))


def _union_checker(*times_of_day):
    return CalendarChecker(
        UnionCalendar.merge(
            [
                TimeOfDayCalendar(TimeOfDay.from_str(time_of_day))
                for time_of_day in times_of_day
            ]
        )
    )


@pytest.mark.parametrize(
    "checker, metadata, target_time_range, expect",
    [
        (
            _time_of_day_checker("13:14:15.16 [America/New_York]"),
            _mock_metadata(TimeRange("2000-01-01", "2022-06-30"), None),
            TimeRange("2001", "2002"),
            True,
        ),
        (
            _time_of_day_checker("13:14:15.16 [America/New_York]"),
            _mock_metadata(None, None, False),
            TimeRange("2001", "2002"),
            False,
        ),
        (
            _time_of_day_checker("13:14:15.16 [America/New_York]"),
            _mock_metadata(TimeRange("2000-01-01", "2022-06-30"), None),
            TimeRange("2001", "2022-06-30 13:14:15.16 [UTC]"),
            True,
        ),
        (  # time ranges are open at the end, so a time range ending at a target
            # point does not include the target point.
            _time_of_day_checker("13:14:15.16 [America/New_York]"),
            _mock_metadata(TimeRange("2000-01-01", "2022-06-30"), None),
            TimeRange("2001", "2022-06-30 13:14:15.16 [America/New_York]"),
            True,
        ),
        (
            _time_of_day_checker("13:14:15.16 [America/New_York]"),
            _mock_metadata(TimeRange("2000-01-01", "2022-06-30"), None),
            TimeRange("2001", "2022-06-30 13:14:15.17 [America/New_York]"),
            False,
        ),
        (
            IrregularChecker(),
            _mock_metadata(
                None, TimeRange("2019", "2021")  # data time range ignored for irregular
            ),
            TimeRange("2020", "2021"),
            True,
        ),
        (
            IrregularChecker(),
            _mock_metadata(
                None, TimeRange("2019", "2021")  # data time range ignored for irregular
            ),
            TimeRange("2000", "2001"),
            ValueError(
                "Increments should not be run with non-overlapping time ranges."
            ),
        ),
        (
            IrregularChecker(),
            _mock_metadata(
                None, TimeRange("2019", "2021")  # data time range ignored for irregular
            ),
            TimeRange("2020", "2022"),
            False,
        ),
        (IrregularChecker(), _mock_metadata(None, None, False), None, False),
    ],
)
def test_is_complete(checker, metadata, target_time_range, expect):
    assert_call(checker.is_complete, expect, metadata, target_time_range)


def _mock_dependency(checker: ICompletionChecker, is_ts: bool, inherit=None):
    mock = Mock()
    mock.task.time_series = is_ts
    mock.task.completion_checker = checker
    mock.inherit_frequency = inherit
    return mock


@pytest.mark.parametrize(
    "dependencies, expect",
    [
        ({}, ValueError("Task has no dependencies")),
        (
            {
                "foo": _mock_dependency(
                    _time_of_day_checker("13:14:15.16 [America/New_York]"), True
                )
            },
            _time_of_day_checker("13:14:15.16 [America/New_York]"),
        ),
        (
            {
                "foo": _mock_dependency(
                    _time_of_day_checker("13:14:15.16 [America/New_York]"), True
                ),
                "bar": _mock_dependency(
                    _time_of_day_checker("13:14:15.16 [America/New_York]"), True
                ),
            },
            _union_checker(
                "13:14:15.16 [America/New_York]", "13:14:15.16 [America/New_York]"
            ),
        ),
        (
            {
                "foo": _mock_dependency(
                    _time_of_day_checker("17:14:15.16 [America/New_York]"), True
                ),
                "bar": _mock_dependency(
                    _time_of_day_checker("13:14:15.16 [America/New_York]"), True
                ),
            },
            _union_checker(
                "13:14:15.16 [America/New_York]", "17:14:15.16 [America/New_York]"
            ),
        ),
        (
            {
                "foo": _mock_dependency(
                    _union_checker(
                        "13:14:15.16 [America/New_York]",
                        "17:14:15.16 [America/New_York]",
                    ),
                    True,
                ),
                "bar": _mock_dependency(
                    _time_of_day_checker("14:14:15.16 [UTC]"), True
                ),
            },
            _union_checker(
                "13:14:15.16 [America/New_York]",
                "17:14:15.16 [America/New_York]",
                "14:14:15.16 [UTC]",
            ),
        ),
        ({"foo": _mock_dependency(IrregularChecker(), True)}, IrregularChecker()),
        (
            {
                "foo": _mock_dependency(IrregularChecker(), True),
                "bar": _mock_dependency(None, False),
            },
            IrregularChecker(),
        ),
        (
            {
                "foo": _mock_dependency(IrregularChecker(), True),
                "bar": _mock_dependency(IrregularChecker(), True),
            },
            IrregularChecker(),
        ),
        (
            {
                "foo": _mock_dependency(IrregularChecker(), True),
                "bar": _mock_dependency(
                    _time_of_day_checker("14:14:15.16 [UTC]"), True
                ),
            },
            ValueError("Task has inconsistent completion checkers"),
        ),
        (
            {
                "foo": _mock_dependency(IrregularChecker(), True),
                "bar": _mock_dependency(
                    _time_of_day_checker("14:14:15.16 [UTC]"), True, inherit=False
                ),
            },
            IrregularChecker(),
        ),
        (
            {
                "foo": _mock_dependency(
                    IrregularChecker(),
                    True,
                    inherit=True,
                ),
                "bar": _mock_dependency(
                    _time_of_day_checker("14:14:15.16 [UTC]"),
                    True,
                ),
            },
            IrregularChecker(),
        ),
    ],
)
def test_infer_inherited_completion_checker(dependencies, expect):
    assert_call(infer_inherited_completion_checker, expect, dependencies)
