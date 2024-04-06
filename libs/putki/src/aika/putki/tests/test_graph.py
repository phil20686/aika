import pytest
from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
import pandas as pd

from aika.putki import IrregularChecker
from aika.putki.context import GraphContext, Defaults
from aika.putki.graph import Graph
from aika.time import TimeRange


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


@pytest.fixture()
def context() -> GraphContext:
    file_system_backend = HashBackedPersistanceEngine()
    return GraphContext(
        defaults=Defaults(
            version="0.0.1",
            time_range=TimeRange("2000", "2001"),
            persistence_engine=file_system_backend,
        ),
        namespace="testing",
    )


class TestGraph:
    def test_one(self, context):
        grandparent = context.time_series_task(
            "grandparent",
            _return_random_data,
            foo=10,
            bar=5,
            completion_checker=IrregularChecker(),
        )
        parent = context.time_series_task(
            "parent",
            _return_random_data,
            foo=grandparent,
            bar=7,
        )

        child = context.time_series_task(
            "child", _return_random_data, foo=grandparent, bar=parent
        )

        graph_all = Graph([child])
        graph_parent = Graph([parent])
        graph_isolated = Graph([grandparent])

        assert len(graph_all.all_tasks) == 3
        assert len(graph_parent.all_tasks) == 2
        assert len(graph_isolated.all_tasks) == 1
        assert len(graph_all.sinks) == 1
        assert len(graph_parent.sinks) == 1
        assert len(graph_isolated.sinks) == 1
