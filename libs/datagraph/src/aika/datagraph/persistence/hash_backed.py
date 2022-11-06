import re
import typing as t
from abc import abstractmethod

from frozendict import frozendict
from overrides import overrides
from typing_extensions import Protocol

from aika.datagraph.interface import (
    DataSet,
    DataSetMetadata,
    DataSetMetadataStub,
    IPersistenceEngine,
)
from aika.time.time_range import TimeRange


class HashBackedPersistanceEngine(IPersistenceEngine):
    """
    This is a purely in-memory storage engine, backed by a dictionary, and only suitable
    for single-threaded use and testing.
    """

    _cache: t.Dict[DataSetMetadata, DataSet]

    def __init__(self):
        self._cache = {}

    @overrides()
    def set_state(self) -> t.Dict[str, t.Any]:
        raise ValueError("Cannot persist an in-memory engine")  # pragma: no cover

    @overrides()
    def exists(self, metadata: DataSetMetadata) -> bool:
        return metadata in self._cache

    @overrides()
    def get_predecessors_from_hash(
        self, name: str, hash: int
    ) -> frozendict[str, DataSetMetadataStub]:
        for metadata in self._cache.keys():
            if metadata.__hash__() == hash and metadata.name == name:
                return frozendict(
                    {
                        key: DataSetMetadataStub(
                            name=meta.name,
                            time_level=meta.time_level,
                            static=meta.static,
                            version=meta.version,
                            engine=meta.engine,
                            params=meta.params,
                            hash=meta.__hash__(),
                        )
                        for key, meta in metadata.predecessors.items()
                    }
                )
        raise ValueError("No Matching DataSet")

    @overrides()
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

    @overrides()
    def read(self, metadata: DataSetMetadata, time_range: t.Optional[TimeRange] = None):
        dataset = self.get_dataset(metadata, time_range=time_range)
        if dataset is None:
            return None
        else:
            return dataset.data

    @overrides()
    def get_data_time_range(self, metadata: DataSetMetadata) -> t.Optional[TimeRange]:
        if metadata.static:
            raise ValueError("Cannot get data time range for static dataset")

        dataset = self._cache.get(metadata)
        if dataset is None:
            return None
        else:
            return TimeRange.from_pandas(dataset.data, level=metadata.time_level)

    @overrides()
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

    @overrides()
    def idempotent_insert(
        self,
        dataset: DataSet,
    ) -> bool:
        exists = self.exists(dataset.metadata)
        if not exists:
            self._cache[dataset.metadata] = dataset
        return exists

    @overrides()
    def replace(
        self,
        dataset: DataSet,
    ) -> bool:
        original_size = len(self._cache)
        self._cache[dataset.metadata] = dataset
        return original_size == len(self._cache)

    @overrides()
    def append(
        self,
        dataset: DataSet,
    ) -> bool:
        if dataset.metadata.static:
            raise ValueError("Can only append for time-series data")
        return self._combine_insert(dataset, combine_method=self._append)

    @overrides()
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

        old_dataset = self._cache.get(dataset.metadata)
        if old_dataset is None:
            self._cache[dataset.metadata] = dataset
            return False

        else:

            final_dataset = combine_method(existing=old_dataset, new=dataset)
            self._cache[final_dataset.metadata] = final_dataset
            return True

    @overrides()
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

    @overrides()
    def delete(self, metadata: DataSetMetadata, recursive=False):
        if recursive:
            for successor in self.find_successors(metadata):
                self.delete(successor, recursive=True)
        return self._delete_leaf(metadata)

    @overrides()
    def find(self, match: str, version: t.Optional[str] = None) -> t.List[str]:
        names = []
        for metadata in self._cache.keys():
            if re.match(match, metadata.name) and (
                (version is None) or (version == metadata.version)
            ):
                names.append(metadata.name)
        return list(sorted(names))

    @overrides()
    def scan(
        self, dataset_name: str, params: t.Optional[t.Dict] = None
    ) -> t.Set[DataSetMetadataStub]:
        results = set()
        for metadata in self._cache.keys():
            if metadata.name == dataset_name and (
                not params
                or all(
                    [
                        metadata.recursively_get_parameter_value(target) == value
                        for target, value in params.items()
                    ]
                )
            ):
                results.add(metadata)
        return results
