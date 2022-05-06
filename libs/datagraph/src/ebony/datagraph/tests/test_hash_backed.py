from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from ebony.time.tests.utils import assert_equal
from ebony.time.timestamp import Timestamp

from ebony.datagraph.interface import DataSet
from ebony.datagraph.persistence.hash_backed import HashBackedPersistanceEngine

leaf1 = DataSet.build(
    name="leaf1",
    data=pd.DataFrame(
        1.0,
        columns=list("ABC"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"foo": 1.0, "bar": "baz"},
    predecessors={},
)


def _insert_nans(data: pd.DataFrame, locations: List[Tuple]):
    data = data.copy()
    for i, j in locations:
        data.iloc[i, j] = np.nan
    return data


leaf1_with_nan = leaf1.update(
    _insert_nans(leaf1.data, [(1, 1), (2, 2)]), leaf1.declared_time_range
)

leaf1_extended = DataSet.build(
    name="leaf1",
    data=pd.DataFrame(
        1.1,
        columns=list("ABC"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=12)],
    ),
    params={"foo": 1.0, "bar": "baz"},
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
    predecessors={},
)

child = DataSet.build(
    name="child",
    data=pd.DataFrame(
        2.0,
        columns=list("XZY"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"bananas": "some", "apples": 3.0},
    predecessors={"foo": leaf1.metadata, "bar": leaf2.metadata},
)


@pytest.mark.parametrize(
    "list_of_datasets, expected_final_cache",
    [
        (
            [
                leaf1,
            ],
            {leaf1: leaf1},
        ),
        (
            [
                leaf1,
                leaf2,
                child,
            ],
            {leaf1.metadata: leaf1, leaf2.metadata: leaf2, child.metadata: child},
        ),
        (
            [
                leaf1_with_nan,
                leaf1_extended,
            ],
            {
                leaf1_final.metadata: leaf1_final.update(
                    _insert_nans(leaf1_final.data, [(1, 1), (2, 2)]),
                    leaf1_final.declared_time_range,
                )
            },
        ),
        (
            [
                leaf1,
                leaf1_extended,
            ],
            {leaf1_final.metadata: leaf1_final},
        ),
    ],
)
def test_append(list_of_datasets, expected_final_cache):
    engine = HashBackedPersistanceEngine()
    new_expectation = {}
    for dataset in expected_final_cache.values():
        dataset = DataSet.replace_engine(dataset, engine)
        new_expectation[dataset.metadata] = dataset

    for dataset in list_of_datasets:
        dataset = dataset.replace_engine(dataset, engine)
        engine.append(dataset)

    assert_equal(engine._cache, new_expectation)

    for metadata in new_expectation.keys():
        assert engine.exists(metadata)


@pytest.mark.parametrize(
    "list_of_datasets, expected_final_cache",
    [
        (
            [
                leaf1,
            ],
            {leaf1: leaf1},
        ),
        (
            [
                leaf1,
                leaf2,
                child,
            ],
            {leaf1.metadata: leaf1, leaf2.metadata: leaf2, child.metadata: child},
        ),
        (
            [
                leaf1_with_nan,
                leaf1_extended,
            ],
            {
                leaf1_final.metadata: leaf1_final.update(
                    leaf1_with_nan.data.combine_first(leaf1_extended.data),
                    leaf1_final.declared_time_range,
                )
            },
        ),
        (
            [
                leaf1,
                leaf1_extended,
            ],
            {leaf1_final.metadata: leaf1_final},
        ),
    ],
)
def test_merge(list_of_datasets, expected_final_cache):
    engine = HashBackedPersistanceEngine()
    new_expectation = {}
    for dataset in expected_final_cache.values():
        dataset = DataSet.replace_engine(dataset, engine)
        new_expectation[dataset.metadata] = dataset

    for dataset in list_of_datasets:
        dataset = dataset.replace_engine(dataset, engine)
        engine.merge(dataset)

    assert_equal(engine._cache, new_expectation)

    for metadata in new_expectation.keys():
        assert engine.exists(metadata)
