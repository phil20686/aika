import mongomock as mongomock
import pymongo
import pytest as pytest

from aika.time.tests.utils import assert_equal

from aika.datagraph.interface import DataSet
from aika.datagraph.persistence.mongo_backed import MongoBackedPersistanceEngine
from aika.datagraph.tests.persistence_tests import append_tests, merge_tests


@mongomock.patch()
@pytest.mark.parametrize("list_of_datasets, expected_contents", append_tests)
def test_append(list_of_datasets, expected_contents):
    engine = MongoBackedPersistanceEngine(
        client=pymongo.MongoClient(), database_name="foo"
    )
    engine._client.drop_database("foo")

    new_expectation = {}
    for dataset in expected_contents.values():
        dataset = DataSet.replace_engine(dataset, engine, include_predecessors=True)
        new_expectation[dataset.metadata] = dataset

    for dataset in list_of_datasets:
        dataset = dataset.replace_engine(dataset, engine, include_predecessors=True)
        engine.append(dataset)

    for metadata in new_expectation.keys():
        assert engine.exists(metadata)

    for expected_dataset in new_expectation.values():
        persisted_dataset = engine.get_dataset(expected_dataset.metadata)
        assert persisted_dataset.metadata == expected_dataset.metadata
        assert hash(persisted_dataset.metadata) == hash(expected_dataset.metadata)
        assert_equal(persisted_dataset.data, expected_dataset.data)
        assert (
            persisted_dataset.declared_time_range
            == expected_dataset.declared_time_range
        )
        assert persisted_dataset.data_time_range == expected_dataset.declared_time_range


@mongomock.patch()
@pytest.mark.parametrize("list_of_datasets, expected_contents", merge_tests)
def test_merge(list_of_datasets, expected_contents):
    engine = MongoBackedPersistanceEngine(
        client=pymongo.MongoClient(), database_name="foo"
    )
    engine._client.drop_database("foo")

    new_expectation = {}
    for dataset in expected_contents.values():
        dataset = DataSet.replace_engine(dataset, engine, include_predecessors=True)
        new_expectation[dataset.metadata] = dataset

    for dataset in list_of_datasets:
        dataset = dataset.replace_engine(dataset, engine, include_predecessors=True)
        engine.merge(dataset)

    for metadata in new_expectation.keys():
        assert engine.exists(metadata)

    for expected_dataset in new_expectation.values():
        persisted_dataset = engine.get_dataset(expected_dataset.metadata)
        assert persisted_dataset.metadata == expected_dataset.metadata
        assert hash(persisted_dataset.metadata) == hash(expected_dataset.metadata)
        assert_equal(persisted_dataset.data, expected_dataset.data)
        assert (
            persisted_dataset.declared_time_range
            == expected_dataset.declared_time_range
        )
        assert persisted_dataset.data_time_range == expected_dataset.declared_time_range
