import pytest
import pandas as pd
from ebony.data_graph.interface import DataSet
from ebony.data_graph.persistence.hash_backed import HashBackedPersistanceEngine
from ebony.time.tests.utils import assert_equal
from ebony.time.timestamp import Timestamp

leaf1 = DataSet(
    name="leaf1",
    data=pd.DataFrame(
        1.0,
        columns=list("ABC"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"foo": 1.0, "bar": "baz"},
    predecessors={},
)


@pytest.mark.parametrize(
    "list_of_datasets, expected_final_cache",
    [
        (
            [
                leaf1,
            ],
            {leaf1: leaf1},
        )
    ],
)
def test_append(list_of_datasets, expected_final_cache):
    engine = HashBackedPersistanceEngine()
    for dataset in list_of_datasets:
        engine.append(dataset)
    assert_equal(engine._cache, expected_final_cache)
