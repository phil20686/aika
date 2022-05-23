from typing import List, Tuple

import numpy as np
import pandas as pd

from aika.time.timestamp import Timestamp

from aika.datagraph.interface import DataSet

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

append_tests = [
    (
        [
            leaf1,
        ],
        {leaf1},
    ),
    (
        [
            leaf1,
            leaf2,
            child,
        ],
        {leaf1, leaf2, child},
    ),
    (
        [
            leaf1_with_nan,
            leaf1_extended,
        ],
        {
            leaf1_final.update(
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
        {leaf1_final},
    ),
]

merge_tests = [
    (
        [
            leaf1,
        ],
        {leaf1},
    ),
    (
        [
            leaf1,
            leaf2,
            child,
        ],
        {leaf1, leaf2, child},
    ),
    (
        [
            leaf1_with_nan,
            leaf1_extended,
        ],
        {
            leaf1_final.update(
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
        {leaf1_final},
    ),
]
# list to insert, metadata to check, expected datasets
find_successors_tests = [
    ([leaf1, leaf2, child], leaf1.metadata, {child.metadata}),
    ([leaf1], leaf2.metadata, set()),
]


# list to insert,
# metadata to delete,
# recursive,
# deletion_expectation,
# remaining_datasets
deletion_tests = [
    ([leaf1, leaf2, child], child.metadata, True, True, {leaf1, leaf2}),
    ([leaf1, leaf2, child], leaf2.metadata, False, ValueError, {leaf1, leaf2, child}),
    ([leaf1, leaf2, child], leaf2.metadata, True, True, {leaf1}),
    ([leaf1], leaf2.metadata, True, False, {leaf1}),
]
