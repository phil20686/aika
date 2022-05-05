import pandas as pd
import pytest

from ebony.time.tests.utils import assert_equal
from ebony.time.timestamp import Timestamp

from ebony.datagraph.interface import DataSet, DataSetDeclaration
from ebony.datagraph.persistence.hash_backed import HashBackedPersistanceEngine

leaf1 = DataSet.build(
    name="leaf1",
    data=pd.DataFrame(
        1.0,
        columns=list("ABC"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"foo": 1.0, "bar": "baz"},
    completeness_checker="Regular",
    predecessors={},
)

leaf1_extended = DataSet.build(
    name="leaf1",
    data=pd.DataFrame(
        1.1,
        columns=list("ABC"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=12)],
    ),
    params={"foo": 1.0, "bar": "baz"},
    completeness_checker="Regular",
    predecessors={},
)

leaf1_final = DataSet.build(
    name="leaf1",
    data=pd.DataFrame(
        [[1.0, 1.0, 1.0]] * 10 + [[1.1, 1.1, 1.1]] * 2,
        columns=list("ABC"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=12)],
    ),
    params={"foo": 1.0, "bar": "baz"},
    completeness_checker="Regular",
    predecessors={},
)

leaf2 = DataSet.build(
    name="leaf2",
    data=pd.DataFrame(
        2.0,
        columns=list("XZY"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"foo": 2.0, "bar": "baz"},
    completeness_checker="Regular",
    predecessors={},
)


@pytest.mark.parametrize(
    "list_of_datasets, expected_final_cache",
    [
        # (
        #     [
        #         leaf1,
        #     ],
        #     {leaf1: leaf1},
        # ),
        (
            [
                leaf1,
                leaf1_extended,
            ],
            {leaf1_final.metadata: leaf1_final},
        )
    ],
)
def test_append(list_of_datasets, expected_final_cache):
    engine = HashBackedPersistanceEngine()
    for dataset in list_of_datasets:
        engine.append(dataset)
    assert_equal(engine._cache, expected_final_cache)
