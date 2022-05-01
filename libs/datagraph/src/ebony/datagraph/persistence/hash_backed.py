import typing as t
from abc import abstractmethod

import pandas as pd
from typing_extensions import Protocol

from ebony.datagraph.interface import (
    IPersistenceEngine,
    DataSet,
    DataSetMetadata,
)
from ebony.time.time_range import TimeRange


class HashBackedPersistanceEngine(IPersistenceEngine):
    """
    This is a purely in-memory storage engine, backed by a dictionary, and only suitable
    for single-threaded use and testing.
    """

    _cache: t.Dict[DataSetMetadata, DataSet]

    def __init__(self):
        self._cache = {}

    def exists(self, metadata: DataSetMetadata) -> bool:
        return metadata in self._cache

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
                    data=time_range.view(result.data),
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

    @staticmethod
    def _append(existing, new):
        new_data = TimeRange(existing.data_time_range.end, None).view(new.data)

        if new_data.empty:
            return existing

        return DataSet(
            metadata=new.metadata,
            data=pd.concat([existing.data, new_data], axis=0),
            declared_time_range=TimeRange(
                existing.declared_time_range.start,
                new.declared_time_range.end,
            ),
        )

    @staticmethod
    def _merge(existing: DataSet, new: DataSet) -> DataSet:
        return DataSet(
            metadata=existing.metadata,
            data=existing.data.combine_first(new.data),
            declared_time_range=existing.declared_time_range.union(
                new.declared_time_range
            ),
        )
