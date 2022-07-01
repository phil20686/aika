import pandas as pd

from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
from aika.time.calendars import TimeOfDayCalendar
from aika.time.time_of_day import TimeOfDay
from aika.time.time_range import TimeRange
from aika.utilities.testing import assert_equal

from aika.putki import CalendarChecker, StaticFunctionWrapper, TimeSeriesFunctionWrapper
from aika.putki.interface import Dependency


def _addition(a, b):
    return a + b


def _input_closure(data):
    def f():
        return data

    return f


class TestStaticFunctionWrapper:
    def test_creation(self):
        data1 = pd.DataFrame(
            1.0,
            pd.date_range("2020-01-01 12:00", freq="D", periods=10, tz="UTC"),
            columns=list("ABC"),
        )
        data2 = pd.DataFrame(
            2.0,
            pd.date_range("2020-01-01 12:00", freq="D", periods=10, tz="UTC"),
            columns=list("ABC"),
        )
        engine = HashBackedPersistanceEngine()
        leaf1 = StaticFunctionWrapper(
            name="leaf1",
            namespace="foo",
            version="0.0.1",
            persistence_engine=engine,
            function=_input_closure(data1),
            scalar_kwargs={},
            dependencies={},
        )
        leaf1.run()
        assert_equal(leaf1.read(), data1)

        child1 = StaticFunctionWrapper(
            name="child1",
            namespace="foo",
            version="0.0.1",
            persistence_engine=engine,
            function=_addition,
            scalar_kwargs={"a": 10.0},
            dependencies={"b": Dependency(leaf1)},
        )
        child1.run()
        assert child1.complete()
        assert_equal(child1.read(), data1.add(10.0))


class TestTimeSeriesFunctionWrapper:
    def test_creation(self):
        data1 = pd.DataFrame(
            1.0,
            pd.date_range("2020-01-01 12:00", freq="D", periods=10, tz="UTC"),
            columns=list("ABC"),
        )
        data2 = pd.DataFrame(
            2.0,
            pd.date_range("2020-01-01 12:00", freq="D", periods=10, tz="UTC"),
            columns=list("ABC"),
        )
        engine = HashBackedPersistanceEngine()
        target_time_range = TimeRange("2020-01-05", "2020-01-11")
        leaf1 = TimeSeriesFunctionWrapper(
            name="leaf1",
            namespace="foo",
            version="0.0.1",
            persistence_engine=engine,
            function=_input_closure(data1),
            time_range=target_time_range,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("12:00 [UTC]"), weekdays=list(range(7))
                )
            ),
            scalar_kwargs={},
            dependencies={},
        )
        leaf1.run()
        assert_equal(leaf1.read(), target_time_range.view(data1))

        child1 = TimeSeriesFunctionWrapper(
            name="child1",
            namespace="foo",
            version="0.0.1",
            persistence_engine=engine,
            function=_addition,
            time_range=target_time_range,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("12:00 [UTC]"), weekdays=list(range(7))
                )
            ),
            scalar_kwargs={"a": 10.0},
            dependencies={"b": Dependency(leaf1)},
        )
        child1.run()
        assert child1.complete()
        assert_equal(child1.read(), target_time_range.view(data1.add(10.0)))

    def test_increments(self):
        data1 = pd.DataFrame(
            1.0,
            pd.date_range("2020-01-01 12:00", freq="D", periods=10, tz="UTC"),
            columns=list("ABC"),
        )
        data2 = pd.DataFrame(
            2.0,
            pd.date_range("2020-01-01 12:00", freq="D", periods=10, tz="UTC"),
            columns=list("ABC"),
        )
        engine = HashBackedPersistanceEngine()
        first_range = TimeRange("2020-01-02", "2020-01-06")
        second_range = TimeRange("2020-01-05", "2020-01-11")

        leaf1 = TimeSeriesFunctionWrapper(
            name="leaf1",
            namespace="foo",
            version="0.0.1",
            persistence_engine=engine,
            function=_input_closure(data1),
            time_range=first_range,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("12:00 [UTC]"), weekdays=list(range(7))
                )
            ),
            scalar_kwargs={},
            dependencies={},
        )
        leaf1.run()
        assert_equal(leaf1.read(), first_range.view(data1))

        child1 = TimeSeriesFunctionWrapper(
            name="child1",
            namespace="foo",
            version="0.0.1",
            persistence_engine=engine,
            function=_addition,
            time_range=first_range,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("12:00 [UTC]"), weekdays=list(range(7))
                )
            ),
            scalar_kwargs={"a": 10.0},
            dependencies={"b": Dependency(leaf1)},
        )
        child1.run()
        assert child1.complete()
        assert_equal(child1.read(), first_range.view(data1.add(10.0)))

        leaf1 = TimeSeriesFunctionWrapper(
            name="leaf1",
            namespace="foo",
            version="0.0.1",
            persistence_engine=engine,
            function=_input_closure(data1),
            time_range=second_range,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("12:00 [UTC]"), weekdays=list(range(7))
                )
            ),
            scalar_kwargs={},
            dependencies={},
        )
        leaf1.run()
        assert_equal(leaf1.read(), first_range.union(second_range).view(data1))

        child1 = TimeSeriesFunctionWrapper(
            name="child1",
            namespace="foo",
            version="0.0.1",
            persistence_engine=engine,
            function=_addition,
            time_range=second_range,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("12:00 [UTC]"), weekdays=list(range(7))
                )
            ),
            scalar_kwargs={"a": 10.0},
            dependencies={"b": Dependency(leaf1)},
        )
        child1.run()
        assert child1.complete()
        assert_equal(
            child1.read(), first_range.union(second_range).view(data1.add(10.0))
        )
