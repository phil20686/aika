from typing import Union

import pandas as pd

from ebony.data_graph.interface import iPersistenceEngine, DataSetDeclaration, DataSet
from ebony.time.time_range import TimeRange


class HashBackedPersistanceEngine(iPersistenceEngine):
    """
    This is a purely `in memory` storage engine, backed by a dictionary, and only suitable for single-threaded use
    and testing.
    """

    def __init__(self):
        self._cache = {}

    def exists(self, declaration: DataSetDeclaration) -> Union[None, DataSet]:
        return self._cache.get(declaration, None)

    def replace(self, dataset: DataSet) -> bool:
        original_size = len(self._cache)
        self._cache[dataset.metadata] = dataset
        return original_size == len(self._cache)

    def append(self, dataset: DataSet):
        old_dataset = self.exists(dataset.metadata)
        if old_dataset is None:
            return self.replace(dataset)
        else:
            new_data = pd.concat(
                [
                    old_dataset.data,
                    TimeRange(old_dataset.metadata.data_time_range.end, None).view(
                        dataset.data
                    ),
                ],
                axis=0,
            )
            new_dataset = dataset.update_data(new_data)
            self._cache[new_dataset.metadata] = new_dataset
            return True

    def merge(self, dataset: DataSet) -> bool:
        old_dataset = self.exists(dataset.metadata)
        if old_dataset is None:
            return self.replace(dataset)
        else:
            new_data = old_dataset.data.combine_first(dataset.data)
            new_dataset = dataset.update_data(new_data)
            self._cache[new_dataset.metadata] = new_dataset
            return True
