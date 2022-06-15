"""
The tests here should be a complete set of interface tests for persistent engines that can be run against
any backend
"""
import multiprocessing
from typing import Dict, List, Set, TypeVar, Union

import mongomock as mongomock
import pymongo
import pytest

from aika.utilities.testing import assert_call, assert_equal

from aika.datagraph.interface import DataSet, DataSetMetadata, IPersistenceEngine
from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
from aika.datagraph.persistence.mongo_backed import MongoBackedPersistanceEngine
from aika.datagraph.tests.persistence_tests import (
    append_tests,
    deletion_tests,
    error_condition_tests,
    find_successors_tests,
    merge_tests,
)


@mongomock.patch()
def _mongo_backend_generator():
    database_name = "foo"
    process_local_name = str(id(multiprocessing.current_process()))
    engine = MongoBackedPersistanceEngine(
        client=pymongo.MongoClient(),
        database_name=database_name,
        collection_name=process_local_name,
    )
    engine._client.get_database(database_name).drop_collection(process_local_name)
    return engine


engine_generators = [
    # HashBackedPersistanceEngine,
    _mongo_backend_generator
]

datasets_typevar = TypeVar(
    "datasets_typevar",
    List[DataSet],
    Dict[DataSetMetadata, DataSet],
    Set[DataSet],
    List[DataSetMetadata],
    Set[DataSetMetadata],
)


def _replace_engine(
    engine: IPersistenceEngine, datasets: datasets_typevar
) -> datasets_typevar:
    if isinstance(datasets, list):
        return [
            dataset.replace_engine(engine, include_predecessors=True)
            for dataset in datasets
        ]
    elif isinstance(datasets, set):
        return set(_replace_engine(engine, list(datasets)))
    else:
        result = {}
        for dataset in datasets.values():
            dataset = dataset.replace_engine(engine, include_predecessors=True)
            result[dataset.metadata] = dataset
        return result


def _assert_engine_contains_expected(engine, expected):
    # helper method to make sure that the persistance engine contains all the expected
    # datasets.
    for expected_dataset in expected:
        assert engine.exists(expected_dataset.metadata)
        persisted_dataset = engine.get_dataset(expected_dataset.metadata)
        assert persisted_dataset.metadata == expected_dataset.metadata
        assert hash(persisted_dataset.metadata) == hash(expected_dataset.metadata)
        assert_equal(persisted_dataset.data, expected_dataset.data)
        assert (
            persisted_dataset.declared_time_range
            == expected_dataset.declared_time_range
        )
        assert persisted_dataset.data_time_range == expected_dataset.declared_time_range


@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, expected", append_tests)
def test_append(engine_generator, datasets, expected):
    engine = engine_generator()
    expected = _replace_engine(engine, expected)
    datasets = _replace_engine(engine, datasets)

    for dataset in datasets:
        engine.append(dataset)

    _assert_engine_contains_expected(engine, expected)


@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, expected", merge_tests)
def test_merge(engine_generator, datasets, expected):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    expected = _replace_engine(engine, expected)

    for dataset in datasets:
        engine.merge(dataset)

    _assert_engine_contains_expected(engine, expected)


@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, metadata, expected", find_successors_tests)
def test_find_successors(engine_generator, datasets, metadata, expected):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    metadata = metadata.replace_engine(engine, include_predecessors=True)
    for dataset in datasets:
        engine.idempotent_insert(dataset)

    assert_call(engine.find_successors, _replace_engine(engine, expected), metadata)
    assert_call(engine.find_successors, _replace_engine(engine, expected), metadata)


@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize(
    "datasets, metadata, recursive, deletion_expected, remaining_datasets",
    deletion_tests,
)
def test_delete(
    engine_generator,
    datasets,
    metadata,
    recursive,
    deletion_expected,
    remaining_datasets,
):

    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    metadata = metadata.replace_engine(engine, include_predecessors=True)
    remaining_datasets = _replace_engine(engine, remaining_datasets)

    for dataset in datasets:
        engine.idempotent_insert(dataset)

    assert_call(engine.delete, deletion_expected, metadata, recursive)
    _assert_engine_contains_expected(engine, remaining_datasets)


@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize(
    "datasets_to_insert, func_name, func_kwargs, expect",
    error_condition_tests,
)
def test_error_conditions(
    engine_generator, datasets_to_insert, func_name, func_kwargs, expect
):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets_to_insert)
    for dataset in datasets:
        engine.idempotent_insert(dataset)
    if "metadata" in func_kwargs:
        func_kwargs["metadata"] = func_kwargs["metadata"].replace_engine(engine)
    func = getattr(engine, func_name)
    assert_call(func, expect, **func_kwargs)
