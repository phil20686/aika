"""
The tests here should be a complete set of interface tests for persistent engines that can be run against
any backend
"""
import multiprocessing
import pickle
from typing import Dict, List, Set, TypeVar

import mongomock as mongomock
import pandas as pd
import pytest
from frozendict import frozendict
from mongomock.gridfs import enable_gridfs_integration

from aika.datagraph.interface import (
    DataSet,
    DataSetMetadata,
    DataSetMetadataStub,
    IPersistenceEngine,
)
from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
from aika.datagraph.persistence.mongo_backed import (
    MongoBackedPersistanceEngine,
    UnsecuredLocalhostClient,
)
from aika.datagraph.tests.persistence_tests import (
    append_tests,
    deletion_tests,
    error_condition_tests,
    find_successors_tests,
    find_tests,
    get_dataset_tests,
    idempotent_insert_tests,
    merge_tests,
    param_fidelity_tests,
    predecessor_from_hash_tests,
    replace_tests,
    scan_tests,
)
from aika.utilities.testing import assert_call, assert_equal

enable_gridfs_integration()


def _mongo_backend_generator():
    database_name = "foo"
    process_local_name = str(id(multiprocessing.current_process()))
    client_creator = UnsecuredLocalhostClient()
    engine = MongoBackedPersistanceEngine(
        client_creator=client_creator,
        database_name=database_name,
        collection_name=process_local_name,
    )
    engine._client.get_database(database_name).drop_collection(process_local_name)
    return engine


engine_generators = [HashBackedPersistanceEngine, _mongo_backend_generator]

datasets_typevar = TypeVar(
    "datasets_typevar",
    List[DataSet],
    Dict[DataSetMetadata, DataSet],
    Set[DataSet],
    List[DataSetMetadata],
    Set[DataSetMetadata],
)


def _assert_stub_equals_real(stub: DataSetMetadataStub, metadata: DataSetMetadata):
    assert_equal(
        stub.__hash__(),
        metadata.__hash__(),
    )
    assert_equal(stub.engine, metadata.engine)
    assert_equal(stub.name, metadata.name)
    assert_equal(stub.static, metadata.static)
    assert_equal(stub.time_level, metadata.time_level)
    assert_equal(stub.params, metadata.params)
    assert_equal(stub.predecessors, metadata.predecessors)


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
        raise ValueError("Cannot replace unrecognised type")  # pragma: no cover


def _assert_engine_contains_expected(engine, expected):
    # helper method to make sure that the persistance engine contains all the expected
    # datasets.
    for expected_dataset in expected:
        assert engine.exists(expected_dataset.metadata)
        assert expected_dataset.metadata.exists()
        persisted_dataset = engine.get_dataset(expected_dataset.metadata)
        assert persisted_dataset.metadata == expected_dataset.metadata
        assert hash(persisted_dataset.metadata) == hash(expected_dataset.metadata)
        assert_equal(persisted_dataset.data, expected_dataset.data)
        assert_equal(expected_dataset.metadata.read(), expected_dataset.data)
        if persisted_dataset.metadata.static:
            assert (
                persisted_dataset.declared_time_range
                == expected_dataset.declared_time_range
            )
            assert (
                persisted_dataset.data_time_range
                == expected_dataset.declared_time_range
            )
        if not persisted_dataset.metadata.static:
            assert (
                persisted_dataset.declared_time_range
                == expected_dataset.declared_time_range
                == expected_dataset.metadata.get_declared_time_range()
            )
            assert (
                persisted_dataset.data_time_range
                == expected_dataset.declared_time_range
                == expected_dataset.metadata.get_data_time_range()
            )


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, expected", append_tests)
def test_append(engine_generator, datasets, expected):
    engine = engine_generator()
    expected = _replace_engine(engine, expected)
    datasets = _replace_engine(engine, datasets)

    for dataset in datasets:
        engine.append(dataset)

    _assert_engine_contains_expected(engine, expected)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, expected", append_tests)
def test_append_via_metadata(engine_generator, datasets, expected):
    engine = engine_generator()
    expected = _replace_engine(engine, expected)
    datasets = _replace_engine(engine, datasets)

    for dataset in datasets:
        dataset.metadata.append(
            declared_time_range=dataset.declared_time_range, data=dataset.data
        )

    _assert_engine_contains_expected(engine, expected)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, expected", merge_tests)
def test_merge(engine_generator, datasets, expected):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    expected = _replace_engine(engine, expected)

    for dataset in datasets:
        engine.merge(dataset)

    _assert_engine_contains_expected(engine, expected)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, expected", merge_tests)
def test_merge_via_dataset(engine_generator, datasets, expected):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    expected = _replace_engine(engine, expected)

    for dataset in datasets:
        dataset.metadata.merge(
            declared_time_range=dataset.declared_time_range, data=dataset.data
        )

    _assert_engine_contains_expected(engine, expected)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, replacement, expected", replace_tests)
def test_replace(engine_generator, datasets, replacement, expected):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    replacement = replacement.replace_engine(engine, include_predecessors=True)
    expected = _replace_engine(engine, expected)

    for dataset in datasets:
        engine.idempotent_insert(dataset)

    engine.replace(replacement)

    _assert_engine_contains_expected(engine, expected)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, replacement, expected", replace_tests)
def test_replace_via_metadata(engine_generator, datasets, replacement, expected):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    replacement = replacement.replace_engine(engine, include_predecessors=True)
    expected = _replace_engine(engine, expected)

    for dataset in datasets:
        engine.idempotent_insert(dataset)

    replacement.metadata.replace(
        declared_time_range=replacement.declared_time_range, data=replacement.data
    )

    _assert_engine_contains_expected(engine, expected)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize(
    "datasets, target, expected_predecessors", predecessor_from_hash_tests
)
def test_get_predecessors_from_hash(
    engine_generator, datasets, target, expected_predecessors
):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    target = target.replace_engine(engine, include_predecessors=True)
    expected_predecessors = frozendict(
        {
            name: value.replace_engine(engine)
            for name, value in expected_predecessors.items()
        }
    )

    for dataset in datasets:
        engine.append(dataset)

    result = engine.get_predecessors_from_hash(target.name, target.__hash__())
    # note that this is only required because of the mongomock situation in this test.
    # they natively create new connections but in this case that means that they cannot "see" the contents of
    # mongo mock so we need to put back in the engine that actually contains the data.
    for r in result.values():
        r._engine = engine

    for name, expected in expected_predecessors.items():
        _assert_stub_equals_real(result[name], expected)


@mongomock.patch()
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


@mongomock.patch()
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


@mongomock.patch()
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
    for name in ["metadata", "dataset"]:
        if name in func_kwargs:
            func_kwargs[name] = func_kwargs[name].replace_engine(engine)

    func = getattr(engine, func_name)
    assert_call(func, expect, **func_kwargs)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("input_params, output_params", param_fidelity_tests)
def test_parameter_fidelity(input_params, output_params, engine_generator):
    engine = engine_generator()
    metadata = DataSetMetadata(
        name="test",
        static=True,
        version="foo",
        predecessors={},
        engine=engine,
        params=input_params,
    )
    # assert making a dataset gets the expected result.
    assert_equal(metadata.params, output_params)
    assert_equal(hash(metadata.params), hash(output_params))

    # assert that data with this params can be written
    metadata.idempotent_insert(
        None,
        pd.DataFrame(1.0, columns=list("ABC"), index=range(10)),
    )
    # and then recovered
    new_dataset = metadata.get_dataset(None)
    assert_equal(new_dataset.metadata.params, output_params)
    assert_equal(hash(new_dataset.metadata.params), hash(output_params))


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize(
    "datasets, target, time_range, expected_data", get_dataset_tests
)
def test_get_dataset(engine_generator, datasets, target, time_range, expected_data):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    target = target.replace_engine(engine, include_predecessors=True)
    for dataset in datasets:
        engine.idempotent_insert(dataset)

    result = engine.get_dataset(target, time_range)
    assert_equal(target, result.metadata)
    assert_equal(result.data, expected_data)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, expected", idempotent_insert_tests)
def test_idempotent_insert(engine_generator, datasets, expected):
    engine = engine_generator()
    expected = _replace_engine(engine, expected)
    datasets = _replace_engine(engine, datasets)

    for dataset in datasets:
        engine.idempotent_insert(dataset)

    _assert_engine_contains_expected(engine, expected)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, pattern, version, expected", find_tests)
def test_find(engine_generator, datasets, pattern, version, expected):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)

    for dataset in datasets:
        engine.idempotent_insert(dataset)

    assert_call(engine.find, expected, pattern, version=version)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize("datasets, dataset_name, params, expected", scan_tests)
def test_scan(engine_generator, datasets, dataset_name, params, expected):
    engine = engine_generator()
    datasets = _replace_engine(engine, datasets)
    if isinstance(expected, set):
        # filter out exception cases.
        expected = _replace_engine(engine, expected)

    for dataset in datasets:
        engine.idempotent_insert(dataset)
    assert_call(engine.scan, expected, dataset_name, params)


@mongomock.patch()
@pytest.mark.parametrize("engine_generator", engine_generators)
@pytest.mark.parametrize(
    "method", [x for x in IPersistenceEngine.__dict__ if not x.startswith("_")]
)
def test_docstrings_exist(engine_generator, method):
    engine = engine_generator()
    assert hasattr(engine, method)
    assert getattr(engine, method).__doc__


@mongomock.patch()
def test_mongo_engine_pickling():
    mongo_engine = _mongo_backend_generator()
    new_mongo_engine = pickle.loads(pickle.dumps(mongo_engine))
    assert mongo_engine == new_mongo_engine
