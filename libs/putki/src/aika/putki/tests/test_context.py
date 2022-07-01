from unittest.mock import Mock

import pandas as pd
import pytest as pytest
from pandas.tseries.offsets import BDay

from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
from aika.time.calendars import BUSINESS_DAYS, TimeOfDayCalendar
from aika.time.time_of_day import TimeOfDay
from aika.time.time_range import TimeRange
from aika.utilities.testing import assert_call

from aika.putki import CalendarChecker, ICompletionChecker, IrregularChecker
from aika.putki.context import Defaults, GraphContext, Inference
from aika.putki.tests.test_completion_checking import (
    _time_of_day_checker,
    _union_checker,
)


def _mock_dependency_task(**kwargs):
    mock = Mock()
    for kwarg, value in kwargs.items():
        setattr(mock.task, kwarg, value)
    return mock


def _mock_dependency_checker(
    completion_checker: ICompletionChecker, time_series: bool, inherit_frequency=None
):
    mock = Mock()
    mock.task.time_series = time_series
    mock.task.completion_checker = completion_checker
    mock.inherit_frequency = inherit_frequency
    return mock


class TestInference:
    @pytest.mark.parametrize(
        "dependencies, expect",
        [
            ({}, ValueError("Task has no dependencies")),
            (
                {
                    "foo": _mock_dependency_checker(
                        _time_of_day_checker("13:14:15.16 [America/New_York]"), True
                    )
                },
                _time_of_day_checker("13:14:15.16 [America/New_York]"),
            ),
            (
                {
                    "foo": _mock_dependency_checker(
                        _time_of_day_checker("13:14:15.16 [America/New_York]"), True
                    ),
                    "bar": _mock_dependency_checker(
                        _time_of_day_checker("13:14:15.16 [America/New_York]"), True
                    ),
                },
                _union_checker(
                    "13:14:15.16 [America/New_York]", "13:14:15.16 [America/New_York]"
                ),
            ),
            (
                {
                    "foo": _mock_dependency_checker(
                        _time_of_day_checker("17:14:15.16 [America/New_York]"), True
                    ),
                    "bar": _mock_dependency_checker(
                        _time_of_day_checker("13:14:15.16 [America/New_York]"), True
                    ),
                },
                _union_checker(
                    "13:14:15.16 [America/New_York]", "17:14:15.16 [America/New_York]"
                ),
            ),
            (
                {
                    "foo": _mock_dependency_checker(
                        _union_checker(
                            "13:14:15.16 [America/New_York]",
                            "17:14:15.16 [America/New_York]",
                        ),
                        True,
                    ),
                    "bar": _mock_dependency_checker(
                        _time_of_day_checker("14:14:15.16 [UTC]"), True
                    ),
                },
                _union_checker(
                    "13:14:15.16 [America/New_York]",
                    "17:14:15.16 [America/New_York]",
                    "14:14:15.16 [UTC]",
                ),
            ),
            (
                {"foo": _mock_dependency_checker(IrregularChecker(), True)},
                IrregularChecker(),
            ),
            (
                {
                    "foo": _mock_dependency_checker(IrregularChecker(), True),
                    "bar": _mock_dependency_checker(None, False),
                },
                IrregularChecker(),
            ),
            (
                {
                    "foo": _mock_dependency_checker(IrregularChecker(), True),
                    "bar": _mock_dependency_checker(IrregularChecker(), True),
                },
                IrregularChecker(),
            ),
            (
                {
                    "foo": _mock_dependency_checker(IrregularChecker(), True),
                    "bar": _mock_dependency_checker(
                        _time_of_day_checker("14:14:15.16 [UTC]"), True
                    ),
                },
                ValueError("Task has inconsistent completion checkers"),
            ),
            (
                {
                    "foo": _mock_dependency_checker(IrregularChecker(), True),
                    "bar": _mock_dependency_checker(
                        _time_of_day_checker("14:14:15.16 [UTC]"),
                        True,
                        inherit_frequency=False,
                    ),
                },
                IrregularChecker(),
            ),
            (
                {
                    "foo": _mock_dependency_checker(
                        IrregularChecker(),
                        True,
                        inherit_frequency=True,
                    ),
                    "bar": _mock_dependency_checker(
                        _time_of_day_checker("14:14:15.16 [UTC]"),
                        True,
                    ),
                },
                IrregularChecker(),
            ),
        ],
    )
    def test_infer_inherited_completion_checker(self, dependencies, expect):
        assert_call(Inference.completion_checker, expect, dependencies)

    @pytest.mark.parametrize(
        "predecessors, expect",
        [
            ({}, TimeRange("2020", "2021")),
            (
                {
                    "foo": _mock_dependency_task(
                        time_series=True, time_range=TimeRange("2020", "2021")
                    )
                },
                TimeRange("2020", "2021"),
            ),
            (
                {
                    "foo": _mock_dependency_task(
                        time_series=True, time_range=TimeRange("2020-06-01", "2021")
                    )
                },
                TimeRange("2020-06-01", "2021"),
            ),
            (
                {
                    "foo": _mock_dependency_task(
                        time_series=True,
                        time_range=TimeRange("2020-06-01", "2021-06-1"),
                    )
                },
                TimeRange("2020-06-01", "2021"),
            ),
            (
                {
                    "foo": _mock_dependency_task(
                        time_series=True,
                        time_range=TimeRange("2020-06-01", "2021-06-1"),
                    ),
                    "bar": _mock_dependency_task(
                        time_series=True,
                        time_range=TimeRange("2020-09-01", "2021-06-1"),
                    ),
                },
                TimeRange("2020-09-01", "2021"),
            ),
            (
                {
                    "foo": _mock_dependency_task(
                        time_series=True,
                        time_range=TimeRange("2020-06-01", "2021-06-1"),
                    ),
                    # ignores a time range parameter on a static task.
                    "bar": _mock_dependency_task(
                        time_series=False,
                        time_range=TimeRange("2020-09-01", "2021-06-1"),
                    ),
                },
                TimeRange("2020-06-01", "2021"),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "defaults", [Defaults(time_range=TimeRange("2020", "2021"))]
    )
    def test_inherit_time_range(self, predecessors, expect, defaults):
        assert_call(Inference.infer_time_range, expect, predecessors, defaults)

    @pytest.mark.parametrize(
        "defaults, param_name",
        [(Defaults(version="0.0.1"), "version")],
    )
    @pytest.mark.parametrize(
        "predecessors, expect",
        [
            ({"foo": _mock_dependency_task(version="0.0.1")}, "0.0.1"),
            (
                {"foo": _mock_dependency_task(version="0.0.0")},
                "0.0.0",  # over rides default
            ),
            (
                {
                    "foo": _mock_dependency_task(version="0.0.2"),
                    "bar": _mock_dependency_task(version="0.0.3"),
                },
                ValueError("This context allows exactly one non-default value for"),
            ),
        ],
    )
    def test_infer_not_default(self, defaults, param_name, predecessors, expect):
        assert_call(
            Inference.infer_not_default,
            expect,
            defaults=defaults,
            param_name=param_name,
            predecessors=predecessors,
        )


def _dummy_data_creation(time_range, columns):
    time_of_day = TimeOfDay.from_str("12:00 [America/New_York]")
    index = time_range.view(
        pd.DatetimeIndex(
            [
                time_of_day.make_timestamp(d)
                for d in pd.bdate_range(
                    start=time_range.start - BDay(),
                    end=time_range.end + BDay(),
                )
            ]
        )
    )
    return pd.DataFrame(
        index=index,
        data={name: range(len(index)) for name in columns},
    )


def _rolling_sum(data, window):
    return data.rolling(window=window).sum()


class TestGraphContext:
    def test_one(self):
        """
        Graphs are quite sprawling and the point of the context is to ease the interconnections
        so we can only test by building graphs.
        """
        engine_one = HashBackedPersistanceEngine()
        context = GraphContext(
            defaults=Defaults(
                version="0.0.1",
                time_range=TimeRange("2000", "2001"),
                persistence_engine=engine_one,
            ),
            namespace="testing",
        )

        data = context.time_series_task(
            "dummy_data",
            _dummy_data_creation,
            columns=list("ABC"),
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("12:00 [America/New_York]"),
                    weekdays=BUSINESS_DAYS,
                ),
            ),
        )

        data.run()
        assert data.complete()
        assert data.output.name == "testing.dummy_data"

        rolling_sum_one = context.time_series_task(
            "rolling",
            _rolling_sum,
            data=data,
            window=10,
        )
        assert rolling_sum_one.completion_checker == data.completion_checker
        assert rolling_sum_one.version == data.version
        assert rolling_sum_one.time_range == data.time_range

        rolling_sum_two = context.time_series_task(
            "rolling",
            _rolling_sum,
            data=data,
            window=20,
        )
        rolling_sum_one.run()
        rolling_sum_two.run()
        assert len(engine_one._cache) == 3

        rolling_sum_three = context.time_series_task(
            "rolling", _rolling_sum, data=data, window=10, version="0.0.2"
        )

        assert rolling_sum_three.version == "0.0.2"
        assert rolling_sum_three.output != rolling_sum_one.output
        rolling_sum_three.run()
        assert len(engine_one._cache) == 4
