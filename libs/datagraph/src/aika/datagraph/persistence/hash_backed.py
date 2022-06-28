import typing as t
from abc import abstractmethod

from frozendict import frozendict
from typing_extensions import Protocol

from aika.time.time_range import TimeRange

from aika.datagraph.interface import (
    DataSet,
    DataSetMetadata,
    DatasetMetadataStub,
    IPersistenceEngine,
)


class HashBackedPersistanceEngine(IPersistenceEngine):
    """
    This is a purely in-memory storage engine, backed by a dictionary, and only suitable
    for single-threaded use and testing.
    """

    _cache: t.Dict[DataSetMetadata, DataSet]

    def __init__(self):
        self._cache = {}

    def set_state(self) -> t.Dict[str, t.Any]:
        raise ValueError("Cannot persist an in-memory engine")

    def exists(self, metadata: DataSetMetadata) -> bool:
        return metadata in self._cache

    def get_predecessors_from_hash(
        self, name: str, hash: int
    ) -> frozendict[str, DatasetMetadataStub]:
        for metadata in self._cache.keys():
            if metadata.__hash__() == hash and metadata.name == name:
                return frozendict(
                    {
                        key: DatasetMetadataStub(
                            name=meta.name,
                            time_level=meta.time_level,
                            static=meta.static,
                            engine=meta.engine,
                            params=meta.params,
                            hash=meta.__hash__(),
                        )
                        for key, meta in metadata.predecessors
                    }
                )
        raise ValueError("No Matching Dataset")

    def get_dataset(
        self,
        metadata: DataSetMetadata,
        time_range: t.Optional[TimeRange] = None,
    ) -> DataSet:
        result = self._cache.get(metadata)

        if time_range is not None:
            if metadata.static:
                raise ValueError("time_range must be null for static datasets")
            else:
                result = result.update(
                    data=time_range.view(result.data, result.metadata.time_level),
                    declared_time_range=time_range.intersection(
                        result.declared_time_range
                    ),
                )

        return result

    def read(self, metadata: DataSetMetadata, time_range: t.Optional[TimeRange] = None):
        dataset = self.get_dataset(metadata, time_range=time_range)
        if dataset is None:
            return None
        else:
            return dataset.data

    def get_data_time_range(self, metadata: DataSetMetadata) -> t.Optional[TimeRange]:
        if metadata.static:
            raise ValueError("Cannot get data time range for static dataset")

        dataset = self._cache.get(metadata)
        if dataset is None:
            return None
        else:
            return TimeRange.from_pandas(dataset.data, level=metadata.time_level)

    def get_declared_time_range(
        self, metadata: DataSetMetadata
    ) -> t.Optional[TimeRange]:
        if metadata.static:
            raise ValueError("Cannot get declared time range for static dataset")

        dataset = self._cache.get(metadata)
        if dataset is None:
            return None
        else:
            return dataset.declared_time_range

    def idempotent_insert(
        self,
        dataset: DataSet,
    ) -> bool:
        exists = self.exists(dataset.metadata)
        if not exists:
            self._cache[dataset.metadata] = dataset
        return exists

    def replace(
        self,
        dataset: DataSet,
    ) -> bool:
        original_size = len(self._cache)
        self._cache[dataset.metadata] = dataset
        return original_size == len(self._cache)

    def append(
        self,
        dataset: DataSet,
    ) -> bool:
        if dataset.metadata.static:
            raise ValueError("Can only append for time-series data")
        return self._combine_insert(dataset, combine_method=self._append)

    def merge(self, dataset: DataSet) -> bool:
        if dataset.metadata.static:
            raise ValueError("Can only merge for time-series data")
        return self._combine_insert(dataset, combine_method=self._merge)

    class _CombineMethod(Protocol):
        @abstractmethod
        def __call__(self, existing: DataSet, new: DataSet) -> DataSet:
            """
            Combine `existing` and `new` datasets, with the same metadata, into one
            final dataset.
            """

    def _combine_insert(
        self,
        dataset: DataSet,
        combine_method: _CombineMethod,
    ) -> bool:
        """
        Insert `dataset`, using `combine_method` to combine with any existing dataset
        if applicable.

        Return True iff an existing dataset was found.
        """
        if dataset.metadata.static:
            raise ValueError("Can only append for time-series data")

        old_dataset = self._cache.get(dataset.metadata)
        if old_dataset is None:
            self._cache[dataset.metadata] = dataset
            return False

        else:

            final_dataset = combine_method(existing=old_dataset, new=dataset)
            self._cache[final_dataset.metadata] = final_dataset
            return True

    def find_successors(self, metadata: DataSetMetadata) -> t.Set[DataSetMetadata]:
        return set(
            (md for md in self._cache.keys() if md.is_immediate_predecessor(metadata))
        )

    def _delete_leaf(self, metadata: DataSetMetadata):

        if not self.exists(metadata):
            return False
        else:
            successors = self.find_successors(metadata)
            if len(successors) > 0:
                raise ValueError("Cannot delete a dataset that still has successors")
            else:
                self._cache.pop(metadata)
                return True

    def delete(self, metadata: DataSetMetadata, recursive=False):
        if recursive:
            for successor in self.find_successors(metadata):
                self.delete(successor, recursive=True)
        return self._delete_leaf(metadata)
