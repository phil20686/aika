import typing as t

from aika.time.time_range import TimeRange

from aika.datagraph.interface import (
    DataSet,
    DataSetMetadata,
    DatasetMetadataStub,
    IPersistenceEngine,
)


class CompositeEngine(IPersistenceEngine):
    """
    This allows you to combine a writeable persistence engine with a read only one.
    In a real production setting this would allow you to "branch" off existing datasets in the graph and only
    write the modified datasets locally, greatly improving run time performance in different settings.
    """

    def __init__(
        self, writeable_engine: IPersistenceEngine, read_only_engine: IPersistenceEngine
    ):
        self._writeable = writeable_engine
        self._read_only = read_only_engine

    def __hash__(self):
        return hash((self._writeable, self._read_only))

    def __eq__(self, other):
        # a dataset with a composite engine should evaluate equal to either of
        # the engines it has internally. So that dataset equality can be correctly
        # established.
        if isinstance(other, IPersistenceEngine):
            return other == self._writeable or other == self._read_only
        else:
            return NotImplemented

    def set_state(self) -> t.Dict[str, t.Any]:
        raise ValueError("Cannot persist a composite engine")

    def exists(self, metadata: DataSetMetadata) -> bool:
        return self._writeable.exists(metadata) or self._read_only.exists(metadata)

    def get_predecessors_from_hash(
        self, name: str, hash: int
    ) -> t.Dict[str, DatasetMetadataStub]:
        try:
            return self._writeable.get_predecessors_from_hash(name, hash)
        except ValueError:
            return self._read_only.get_predecessors_from_hash(name, hash)

    def get_dataset(
        self,
        metadata: DataSetMetadata,
        time_range: t.Optional[TimeRange] = None,
    ) -> DataSet:
        return self._writeable.get_dataset(
            metadata, time_range
        ) or self._read_only.get_dataset(metadata, time_range)

    def read(
        self, metadata: DataSetMetadata, time_range: t.Optional[TimeRange] = None
    ) -> t.Any:
        return self._writeable.read(metadata, time_range) or self._read_only.read(
            metadata, time_range
        )

    def get_data_time_range(self, metadata: DataSetMetadata) -> t.Optional[TimeRange]:
        return self._writeable.get_data_time_range(
            metadata
        ) or self._read_only.get_data_time_range(metadata)

    def get_declared_time_range(
        self, metadata: DataSetMetadata
    ) -> t.Optional[TimeRange]:
        return self._writeable.get_declared_time_range(
            metadata
        ) or self._read_only.get_declared_time_range(metadata)

    def idempotent_insert(
        self,
        dataset: DataSet,
    ) -> bool:
        if self.exists(dataset.metadata):
            return True
        else:
            return self._writeable.idempotent_insert(dataset)

    def replace(
        self,
        dataset: DataSet,
    ) -> bool:
        exists = self.exists(dataset.metadata)
        self._writeable.replace(dataset)
        return exists

    def append(
        self,
        dataset: DataSet,
    ) -> bool:
        existing_dataset = self.get_dataset(dataset.metadata)
        if existing_dataset is None:
            return self._writeable.replace(dataset)
        else:
            new_dataset = self._append(existing_dataset, dataset)
            # we do not want to copy existing dataset from one engine to the other if
            # the append command does not result in any new data.
            if existing_dataset.data_time_range.end < new_dataset.data_time_range.end:
                self._writeable.replace(new_dataset)
            return True

    def merge(
        self,
        dataset: DataSet,
    ) -> bool:
        existing_dataset = self.get_dataset(dataset.metadata)
        if existing_dataset is None:
            return self._writeable.replace(dataset)
        else:
            new_dataset = self._merge(existing_dataset, dataset)
            # we do not want to copy existing dataset from one engine to the other if
            # the merge command does not result in any changes.
            if not existing_dataset.data.equals(new_dataset.data):
                self._writeable.replace(new_dataset)
            return True

    def delete(self, metadata: DataSetMetadata, recursive=False):
        # we can only delete from the writeable engine.
        return self._writeable.delete(metadata, recursive=recursive)

    def find_successors(self, metadata: DataSetMetadata) -> t.Set[DatasetMetadataStub]:
        return self._read_only.find_successors(
            metadata
        ) + self._writeable.find_successors(metadata)
