import copy
import pickle
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from pandas._libs.tslibs.offsets import BDay

from aika.datagraph.persistence.pure_filesystem_backend import (
    FileSystemPersistenceEngine,
)
from aika.putki import CalendarChecker
from aika.putki.context import GraphContext, Defaults
from aika.putki.graph import Graph
from aika.putki.runners import MultiThreadedRunner, SingleThreadedRunner, IGraphRunner
from aika.time import TimeRange, TimeOfDay, TimeOfDayCalendar
from aika.utilities.hashing import session_consistent_hash

try:
    from aika.putki.luigi_runner import LuigiRunner

    luigi_available = True
except ImportError:
    luigi_available = False


class MockDependency:
    def __init__(self, task):
        self.task = task


class MockTask:
    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return session_consistent_hash(self.name)

    def __init__(
        self,
        name,
        is_complete=False,
        dependencies: Dict[str, "MockTask"] = None,
        should_raise=False,
    ):
        self.name = name
        self._is_complete = is_complete
        self._dependencies = (
            {}
            if dependencies is None
            else {k: MockDependency(v) for k, v in dependencies.items()}
        )
        self._should_raise = should_raise

    def run(self):
        if self._should_raise:
            raise ValueError("Task raised an error")
        else:
            self._is_complete = True
            return None

    @property
    def dependencies(self):
        return copy.deepcopy(self._dependencies)

    def complete(self):
        return self._is_complete


class TestRunnersWithMocks:
    """
    We can test the simpler runners with mock tasks that are a bit simpler and
    easier to control, however this will not in general work for more complicated
    runners like luigi.
    """

    runners = [
        SingleThreadedRunner,
        MultiThreadedRunner,
    ]

    child = MockTask(
        "child", dependencies={"foo": MockTask("leaf_one"), "bar": MockTask("leaf_two")}
    )
    child_with_complete_parents = MockTask(
        "child_with_complete_parents",
        dependencies={
            "foo": MockTask("leaf_one", is_complete=True),
            "bar": MockTask("leaf_two", is_complete=True),
        },
    )
    child_with_complete_parent_and_grandparent = MockTask(
        "child_with_complete_parent_and_grandparent",
        dependencies={
            "foo": MockTask(
                "parent",
                is_complete=True,
                dependencies={"bar": MockTask("grandparent", is_complete=True)},
            ),
        },
    )
    grand_child = MockTask(
        "grandchild",
        dependencies={
            "foo": MockTask("leaf_one"),
            "bar": MockTask(
                "child",
                dependencies={"foo": MockTask("leaf_one"), "bar": MockTask("leaf_two")},
            ),
        },
    )

    failed_child = MockTask(
        "failed_child",
        dependencies={"foo": MockTask("leaf_one"), "bar": MockTask("leaf_two")},
        should_raise=True,
    )
    child_of_failure = MockTask(
        "child_of_failure",
        dependencies={
            "foo": MockTask("leaf_one"),
            "bar": MockTask(
                "failed_child",
                dependencies={"foo": MockTask("leaf_one"), "bar": MockTask("leaf_two")},
                should_raise=True,
            ),
        },
    )
    diamond_grandchild_of_failure = MockTask(
        "diamond_grandchild_of_failure",
        dependencies={
            "foo": MockTask(
                "child_of_failure_one",
                dependencies={
                    "foo": MockTask("failed_grand_parent", should_raise=True)
                },
            ),
            "bar": MockTask(
                "child_of_failure_two",
                dependencies={
                    #  this should resolve to the same mock task despite having two instances.
                    "foo": MockTask("failed_grand_parent", should_raise=True)
                },
            ),
        },
    )

    @pytest.mark.parametrize("runner", runners)
    @pytest.mark.parametrize(
        "input_graph, expected_complete, expected_failed, expected_not_run",
        [
            (Graph([child]), 3, 0, 0),
            (Graph([child, child]), 3, 0, 0),
            # in this case the graph never looked at the grandparent because it was blocked by a complete parent.
            (Graph([child_with_complete_parent_and_grandparent]), 2, 0, 0),
            (Graph([child_with_complete_parents]), 3, 0, 0),
            (Graph([diamond_grandchild_of_failure]), 0, 1, 3),
            (Graph([grand_child]), 4, 0, 0),
            (Graph([child, grand_child]), 4, 0, 0),
            (Graph([failed_child]), 2, 1, 0),
            (Graph([child_of_failure]), 2, 1, 1),
        ],
    )
    def test_runner(
        self,
        runner,
        input_graph: Graph,
        expected_complete,
        expected_failed,
        expected_not_run,
    ):
        status = runner.run(
            copy.deepcopy(
                input_graph
            )  # Because the mock tasks are mutable copy them before they run.
        )
        assert len(status.ready) == 0
        assert len(status.complete) == expected_complete
        assert len(status.failed) == expected_failed
        assert len(status.waiting) == 0
        assert len(status.has_failed_predecessor) == expected_not_run
        assert "Graph Status:" in str(status)


def _return_random_data(time_range: TimeRange, foo: float, bar: float):
    return (
        pd.DataFrame(
            data=1.0,
            columns=range(10),
            index=pd.date_range(start=time_range.start, end=time_range.end, freq="B"),
        )
        .multiply(foo)
        .subtract(bar)
    )


def _raises_error(foo, bar):
    raise ValueError("Error For Test")


@pytest.fixture(scope="module")
def base_path() -> Path:
    base = Path(__file__).parent / Path(__file__).stem
    if base.exists():
        shutil.rmtree(base)
    yield base
    if base.exists():
        shutil.rmtree(base)


@pytest.fixture()
def file_path(runner: IGraphRunner, base_path, request) -> Path:
    try:
        full_path = base_path / runner.__name__ / request.node.name
    except AttributeError:
        full_path = base_path / type(runner).__name__ / request.node.name
    if full_path.exists():
        shutil.rmtree(full_path)
    yield full_path
    if full_path.exists():
        shutil.rmtree(full_path)


@pytest.fixture()
def context(file_path) -> GraphContext:
    file_system_backend = FileSystemPersistenceEngine(root_file_path=file_path)
    return GraphContext(
        defaults=Defaults(
            version="0.0.1",
            time_range=TimeRange("2000", "2001"),
            persistence_engine=file_system_backend,
        ),
        namespace="testing",
    )


class TestRunnersWithActualGraphs:
    runners = (
        [
            SingleThreadedRunner,
            MultiThreadedRunner,
            LuigiRunner(use_local_scheduler=True),
        ]
        if luigi_available
        else [
            SingleThreadedRunner,
            MultiThreadedRunner,
        ]
    )

    base = Path(__file__).parent / "runners_test"

    @pytest.mark.parametrize("runner", runners)
    def test_one(self, runner: IGraphRunner, context: GraphContext):
        leaf_one = context.time_series_task(  # = 5
            "leaf_one",
            _return_random_data,
            foo=10,
            bar=5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )
        leaf_two = context.time_series_task(  # = 4
            "leaf_two",
            _return_random_data,
            foo=-1,
            bar=-5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )
        child = context.time_series_task(
            "child",
            _return_random_data,
            foo=leaf_one,
            bar=leaf_two,
        )

        child_pickled = pickle.loads(pickle.dumps(child))
        assert child == child_pickled
        assert child.output == child_pickled.output
        assert child.__hash__() == child_pickled.__hash__()

        status = runner.run(Graph([child]))
        assert len(status.ready) == 0
        assert len(status.failed) == 0
        assert len(status.complete) == 3
        assert len(status.waiting) == 0
        assert len(status.has_failed_predecessor) == 0
        assert child.read().eq(1).all().all()

    @pytest.mark.parametrize("runner", runners)
    def test_two(self, runner: IGraphRunner, context: GraphContext):
        leaf_one = context.time_series_task(  # = 5
            "leaf_one",
            _return_random_data,
            foo=10,
            bar=5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )
        leaf_two = context.time_series_task(  # = 4
            "leaf_two",
            _return_random_data,
            foo=-1,
            bar=-5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )
        child = context.time_series_task(
            "child",
            _return_random_data,
            foo=leaf_one,
            bar=leaf_two,
        )

        status = runner.run(Graph([child, child]))
        assert len(status.ready) == 0
        assert len(status.failed) == 0
        assert len(status.complete) == 3
        assert len(status.waiting) == 0
        assert len(status.has_failed_predecessor) == 0
        assert child.read().eq(1).all().all()

    @pytest.mark.parametrize("runner", runners)
    def test_three(self, runner: IGraphRunner, context: GraphContext):
        grand_parent = context.time_series_task(
            "grand_parent",
            _return_random_data,
            foo=10,
            bar=0,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )  # 10

        parent = context.time_series_task(
            "parent", _return_random_data, foo=1, bar=grand_parent
        )  # -9

        child = context.time_series_task(
            "child", _return_random_data, bar=parent, foo=2
        )  # 11
        grand_parent.run()
        parent.run()
        # the parent being complete will block the grand parent from even being seen.
        # only the child will be run, and the parent will have its completeness checked and then
        # the graph building will terminate without ever attempting to run the grand parent.

        input_graph = Graph([child])
        status = runner.run(input_graph)
        assert len(status.ready) == 0
        assert len(status.failed) == 0
        assert len(status.complete) == 2
        assert len(status.waiting) == 0
        assert len(status.has_failed_predecessor) == 0
        assert child.read().eq(11).all().all()

    @pytest.mark.parametrize("runner", runners)
    def test_four(self, runner: IGraphRunner, context: GraphContext):
        leaf_one = context.time_series_task(  # = 5
            "leaf_one",
            _return_random_data,
            foo=10,
            bar=5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )
        leaf_two = context.time_series_task(  # = 4
            "leaf_two",
            _return_random_data,
            foo=-1,
            bar=-5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )
        child = context.time_series_task(
            "child",
            _return_random_data,
            foo=leaf_one,
            bar=leaf_two,
        )

        leaf_one.run()
        leaf_two.run()

        status = runner.run(Graph([child]))
        assert len(status.ready) == 0
        assert len(status.failed) == 0
        assert len(status.complete) == 3
        assert len(status.waiting) == 0
        assert len(status.has_failed_predecessor) == 0
        assert child.read().eq(1).all().all()

    @pytest.mark.parametrize("runner", runners)
    def test_five(self, runner: IGraphRunner, context: GraphContext):
        failing_grand_parent = context.time_series_task(
            "failing_grand_parent",
            _raises_error,
            foo=1,
            bar=1,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )

        parent_one = context.time_series_task(
            "parent_one", _return_random_data, foo=failing_grand_parent, bar=1
        )

        parent_two = context.time_series_task(
            "parent_two", _return_random_data, foo=failing_grand_parent, bar=2
        )

        child = context.time_series_task(
            "child", _return_random_data, foo=parent_one, bar=parent_two
        )

        status = runner.run(Graph([child]))
        assert len(status.ready) == 0
        assert len(status.failed) == 1
        assert len(status.complete) == 0
        assert len(status.waiting) == 0
        assert len(status.has_failed_predecessor) == 3
        assert child.read() is None

    @pytest.mark.parametrize("runner", runners)
    def test_seven(self, runner: IGraphRunner, context: GraphContext):
        grandparent_one = context.time_series_task(  # = 5
            "grandparent_one",
            _return_random_data,
            foo=10,
            bar=5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )
        grand_parent_two = context.time_series_task(  # = 4
            "grand_parent_two",
            _return_random_data,
            foo=-1,
            bar=-5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )

        parent_one = context.time_series_task(  # 1
            "parent_one",
            _return_random_data,
            foo=grandparent_one,
            bar=grand_parent_two,
        )

        child = context.time_series_task(  # -4
            "child", _return_random_data, foo=parent_one, bar=grandparent_one
        )

        status = runner.run(Graph([child]))
        assert len(status.ready) == 0
        assert len(status.failed) == 0
        assert len(status.complete) == 4
        assert len(status.waiting) == 0
        assert len(status.has_failed_predecessor) == 0
        assert child.read().eq(-4).all().all()

    @pytest.mark.parametrize("runner", runners)
    def test_eight(self, runner: IGraphRunner, context: GraphContext):
        grandparent_one = context.time_series_task(  # = 5
            "grandparent_one",
            _return_random_data,
            foo=10,
            bar=5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )
        grand_parent_two = context.time_series_task(  # = 4
            "grand_parent_two",
            _return_random_data,
            foo=-1,
            bar=-5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )

        parent_one = context.time_series_task(  # 1
            "parent_one",
            _return_random_data,
            foo=grandparent_one,
            bar=grand_parent_two,
        )

        child = context.time_series_task(  # -4
            "child", _return_random_data, foo=parent_one, bar=grandparent_one
        )

        # same as previous test except with redundant leaf tasks specified.
        status = runner.run(Graph([child, parent_one]))
        assert len(status.ready) == 0
        assert len(status.failed) == 0
        assert len(status.complete) == 4
        assert len(status.waiting) == 0
        assert len(status.has_failed_predecessor) == 0
        assert child.read().eq(-4).all().all()

    @pytest.mark.parametrize("runner", runners)
    def test_nine(self, runner: IGraphRunner, context: GraphContext):
        grandparent_one = context.time_series_task(  # = 5
            "grandparent_one",
            _return_random_data,
            foo=10,
            bar=5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )
        grand_parent_two = context.time_series_task(  # = 4
            "grand_parent_two",
            _return_random_data,
            foo=-1,
            bar=-5,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(
                    TimeOfDay.from_str("00:00 [UTC]"),
                    freq=BDay(),
                ),
            ),
        )

        parent_one = context.time_series_task(  # 1
            "parent_one",
            _raises_error,
            foo=grandparent_one,
            bar=grand_parent_two,
        )

        child = context.time_series_task(  # -4
            "child", _return_random_data, foo=parent_one, bar=grandparent_one
        )

        # same as previous test except with redundant leaf tasks specified.
        status = runner.run(Graph([child, parent_one]))
        assert len(status.ready) == 0
        assert len(status.failed) == 1
        assert len(status.complete) == 2
        assert len(status.waiting) == 0
        assert len(status.has_failed_predecessor) == 1
        assert child.read() is None
