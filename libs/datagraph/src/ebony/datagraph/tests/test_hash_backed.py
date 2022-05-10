import pytest

from ebony.time.tests.utils import assert_equal

from ebony.datagraph.interface import DataSet
from ebony.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
from ebony.datagraph.tests.persistence_tests import append_tests, merge_tests


@pytest.mark.parametrize("list_of_datasets, expected_contents", append_tests)
def test_append(list_of_datasets, expected_contents):
    engine = HashBackedPersistanceEngine()
    new_expectation = {}
    for dataset in expected_contents.values():
        dataset = DataSet.replace_engine(dataset, engine)
        new_expectation[dataset.metadata] = dataset

    for dataset in list_of_datasets:
        dataset = dataset.replace_engine(dataset, engine)
        engine.append(dataset)

    assert_equal(engine._cache, new_expectation)

    for metadata in new_expectation.keys():
        assert engine.exists(metadata)


@pytest.mark.parametrize("list_of_datasets, expected_contents", merge_tests)
def test_merge(list_of_datasets, expected_contents):
    engine = HashBackedPersistanceEngine()
    new_expectation = {}
    for dataset in expected_contents.values():
        dataset = DataSet.replace_engine(dataset, engine)
        new_expectation[dataset.metadata] = dataset

    for dataset in list_of_datasets:
        dataset = dataset.replace_engine(dataset, engine)
        engine.merge(dataset)

    assert_equal(engine._cache, new_expectation)

    for metadata in new_expectation.keys():
        assert engine.exists(metadata)
