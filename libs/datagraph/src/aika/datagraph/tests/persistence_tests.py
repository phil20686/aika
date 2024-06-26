from typing import List, Tuple

import numpy as np
import pandas as pd
from frozendict import frozendict

from aika.datagraph.interface import DataSet
from aika.time.time_range import TimeRange
from aika.time.timestamp import Timestamp

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

repeated_leaf1 = DataSet.build(
    name="leaf1",
    data=pd.DataFrame(
        2.0,
        columns=list("ABC"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"foo": 2.0, "bar": "baz"},
    predecessors={},
)

repeated2_leaf1 = DataSet.build(
    name="leaf1",
    data=pd.DataFrame(
        3.0,
        columns=list("ABC"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"foo": 2.0, "bar": "bar"},
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
# make sure that in these cases the declared data time range expands correctly when the additional data
# does not go all the way back to the start.
leaf1_extended_late_start = DataSet.build(
    name="leaf1",
    data=pd.DataFrame(
        1.1,
        columns=list("ABC"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-05", periods=8)],
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

child_with_recursive_params = DataSet.build(
    name="child_with_recursive_params",
    data=pd.DataFrame(
        3.0,
        columns=list("XYZ"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"bananas": [{"foo": 3, "bar": ["apples", 3.0]}]},
    predecessors={"foo": leaf1.metadata, "bar": leaf2.metadata},
)

repeated_child = DataSet.build(
    name="child",
    data=pd.DataFrame(
        2.0,
        columns=list("XZY"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"bananas": "some", "apples": 3.0},
    predecessors={"foo": repeated_leaf1.metadata, "bar": leaf2.metadata},
)

repeated2_child = DataSet.build(
    name="child",
    data=pd.DataFrame(
        2.0,
        columns=list("XZY"),
        index=[Timestamp(x) for x in pd.date_range(start="2021-01-01", periods=10)],
    ),
    params={"bananas": "some", "apples": 4.0},
    predecessors={"foo": repeated2_leaf1.metadata, "bar": leaf2.metadata},
)

static_leaf1 = DataSet.build(
    name="static_leaf1",
    data=pd.DataFrame(np.random.randn(20, 12)),
    static=True,
    params={"rows": 12, "foo": 34.5634},
    predecessors={},
)

extremely_large_static_dataset = DataSet.build(
    name="extremely_large_data",
    data=pd.DataFrame(np.random.randn(10000, 10000)),
    static=True,
    params={},
    predecessors={},
)

# add
# replace
# expect
replace_tests = [([leaf1_extended], leaf1, {leaf1})]

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
    (
        [
            leaf1,
            leaf1_extended_late_start,
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
    (
        [
            leaf1,
            leaf1_extended_late_start,
        ],
        {leaf1_final},
    ),
]
# list to insert, metadata to check, expected datasets
find_successors_tests = [
    ([leaf1, leaf2, child], leaf1.metadata, {child.metadata}),
    (
        [leaf1, leaf2, child_with_recursive_params],
        leaf1.metadata,
        {child_with_recursive_params.metadata},
    ),
    (
        [leaf1, leaf2, child, child_with_recursive_params],
        leaf1.metadata,
        {child.metadata, child_with_recursive_params.metadata},
    ),
    ([leaf1], leaf2.metadata, set()),
    ([leaf1], leaf1.metadata, set()),
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

# list to insert
# function
# func kwargs
# expect
error_condition_tests = [
    (
        [],
        "get_predecessors_from_hash",
        {"name": "foo", "hash": 1, "version": "no_version"},
        ValueError,
    ),
    ([], "get_dataset", {"metadata": leaf1.metadata}, None),
    ([leaf2], "get_dataset", {"metadata": leaf1.metadata}, None),
    ([], "get_dataset", {"metadata": static_leaf1.metadata}, None),
    ([], "append", {"dataset": static_leaf1}, ValueError("Can only append for")),
    ([], "merge", {"dataset": static_leaf1}, ValueError("Can only merge for")),
    (
        [],
        "read",
        {"metadata": static_leaf1.metadata, "time_range": TimeRange(None, None)},
        ValueError,
    ),
    ([], "read", {"metadata": leaf1.metadata}, None),
    ([], "read", {"metadata": static_leaf1.metadata}, None),
    (
        [],
        "read",
        {"metadata": static_leaf1.metadata, "time_range": TimeRange(None, None)},
        ValueError,
    ),
    ([], "get_data_time_range", {"metadata": leaf1.metadata}, None),
    ([], "get_data_time_range", {"metadata": static_leaf1.metadata}, ValueError),
    ([], "get_declared_time_range", {"metadata": leaf1.metadata}, None),
    ([], "get_declared_time_range", {"metadata": static_leaf1.metadata}, ValueError),
]

# datasets_to_insert
# metadata to find
# expected predecessors

predecessor_from_hash_tests = [
    (
        [leaf1, leaf2, child],
        child.metadata,
        {"foo": leaf1.metadata, "bar": leaf2.metadata},
    )
]

# input params
# read params
param_fidelity_tests = [
    (
        {"foo": 1.0},
        frozendict({"foo": 1.0}),
    ),
    (
        {"foo": 1},
        frozendict({"foo": 1}),
    ),
    (
        frozendict({"foo": 1}),
        frozendict({"foo": 1}),
    ),
    ({"foo": list("BAC")}, frozendict({"foo": tuple(list("BAC"))})),
    ({"foo": tuple("BAC")}, frozendict({"foo": tuple("BAC")})),
    (
        {"foo": [(1, 2), [1.0, 2.0]], "bar": ((1, 2), (1.0, 2.0))},
        frozendict({"foo": ((1, 2), (1.0, 2.0)), "bar": ((1, 2), (1.0, 2.0))}),
    ),
    (
        {"foo": {"bar": [1, 2, [3, 4]]}},
        frozendict({"foo": frozendict({"bar": (1, 2, (3, 4))})}),
    ),
]

# datasets to insert
# metadata to get
# time range
# expected data
get_dataset_tests = [
    ([leaf1], leaf1.metadata, None, leaf1.data),
    (
        [leaf1],
        leaf1.metadata,
        TimeRange("2021-01-03 12:00", None),
        leaf1.data.loc["2021-01-04":],
    ),
]

# datasets to insert
# expected
idempotent_insert_tests = [
    ([leaf1, leaf1_extended], {leaf1}),
    ([extremely_large_static_dataset], {extremely_large_static_dataset}),
    ([leaf1, leaf2, child, leaf1_extended], {leaf1, leaf2, child}),
]

# datasets to insert
# pattern
# version
# expected
find_tests = [
    ([leaf1, leaf2, child], "leaf", None, ["leaf1", "leaf2"]),
    (
        [leaf2, leaf1, child],
        "leaf",
        None,
        ["leaf1", "leaf2"],
    ),  # always lexicographically ordered
    ([leaf1, leaf2, child], ".*", None, ["child", "leaf1", "leaf2"]),
    ([leaf1, leaf2, child], "leaf", "fake_version", []),
]

# datasets to insert
# dataset name
# params
# expected
scan_tests = [
    (
        [leaf1, repeated_leaf1, repeated2_leaf1],
        "leaf1",
        None,
        {leaf1.metadata, repeated_leaf1.metadata, repeated2_leaf1.metadata},
    ),
    (
        [
            leaf1,
            repeated_leaf1,
            repeated2_leaf1,
            leaf2,
            child,
            repeated_child,
            repeated2_child,
        ],
        "child",
        None,
        {child.metadata, repeated_child.metadata, repeated2_child.metadata},
    ),
    (
        [
            leaf1,
            repeated_leaf1,
            repeated2_leaf1,
            leaf2,
            child,
            repeated_child,
            repeated2_child,
        ],
        "child",
        {"apples": 4.0},
        {repeated2_child.metadata},
    ),
    (
        [
            leaf1,
            repeated_leaf1,
            repeated2_leaf1,
            leaf2,
            child,
            repeated_child,
            repeated2_child,
        ],
        "child",
        {"foo.foo": 2.0},
        {repeated_child.metadata, repeated2_child.metadata},
    ),
    (
        [
            leaf1,
            repeated_leaf1,
            repeated2_leaf1,
            leaf2,
            child,
            repeated_child,
            repeated2_child,
        ],
        "child",
        {"strawberries": 2.0},
        ValueError("Dataset child has no parameter strawberries"),
    ),
]
