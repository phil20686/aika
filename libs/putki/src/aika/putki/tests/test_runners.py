import copy
from typing import Dict

import pytest

from aika.putki.graph import Graph
from aika.putki.runners import MultiThreadedRunner, SingleThreadedRunner


class MockDependency:
    def __init__(self, task):
        self.task = task


class MockTask:
    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

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


class TestRunners:
    runners = [SingleThreadedRunner, MultiThreadedRunner]

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
