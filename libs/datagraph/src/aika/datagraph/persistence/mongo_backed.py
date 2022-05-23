import pickle
import typing as t

import pymongo
from bson.objectid import ObjectId
from frozendict._frozendict import frozendict

from aika.time.time_range import TimeRange

from aika.datagraph.interface import (
    DataSet,
    DataSetMetadata,
    DatasetMetadataStub,
    IPersistenceEngine,
)


class MongoBackedPersistanceEngine(IPersistenceEngine):
    """
    This is a mongo bocked version of the persistence engine.

    Notes
    -----
    Throughout we use metadata.__hash__() rather than hash(metadata) since we
    will look in the future to make our hashes extremely unique, and hash() truncates
    the output of __hash__().
    """

    def __init__(
        self,
        client: pymongo.MongoClient,
        database_name="datagraph",
        collection_name="default",
    ):
        self._client = client
        self._database_name = database_name
        self._collection_name = collection_name
        self._database = self._client.get_database(database_name)
        self._collection = self._database.get_collection(collection_name)
        self._serialise_mode = "pickle"
        self._hash_equality_sufficient = True

    def __hash__(self):
        return hash((self._database_name, self._serialise_mode))

    def __eq__(self, other):
        if isinstance(other, MongoBackedPersistanceEngine):
            return all(
                (
                    self._database_name == other._database_name,
                    self._serialise_mode == other._serialise_mode,
                )
            )
        else:
            return NotImplemented

    @classmethod
    def _create_engine(cls, d: t.Dict[str, t.Any]):
        client = pymongo.MongoClient()
        return cls(client=client, **d)

    def set_state(self) -> t.Dict[str, t.Any]:
        # /TODO figure out the client and authentication bits later.
        return {
            "type": "mongodb",
            "database_name": self._database_name,
            "collection_name": self._collection_name,
        }

    def _serialise_metadata_as_stub(self, metadata: DataSetMetadata):
        return {
            "name": metadata.name,
            "hash": metadata.__hash__(),
            "time_level": metadata.time_level,
            "static": metadata.static,
            "params": metadata.params,
            "engine": metadata.engine.set_state(),
        }

    def _serialise_metadata(self, metadata: DataSetMetadata):
        return {
            "name": metadata.name,
            "hash": metadata.__hash__(),
            "time_level": metadata.time_level,
            "static": metadata.static,
            "params": metadata.params,
            "engine": metadata.engine.set_state(),
            "predecessors": [
                self._serialise_metadata_as_stub(pred) | {"param_name": name}
                for name, pred in metadata.predecessors.items()
            ],
        }

    def _serialise_data(self, dataset: DataSet):
        return {
            "data": pickle.dumps(dataset.data),
            "declared_time_range": repr(dataset.declared_time_range),
            "data_time_range": repr(dataset.data_time_range),
        }

    def _deserialise_metadata_as_stub(self, record: t.Dict):
        return DatasetMetadataStub(
            name=record["name"],
            static=record["static"],
            params=record["params"],
            hash=record["hash"],
            time_level=record["hash"],
            engine=IPersistenceEngine.create_engine(record["engine"]),
        )

    def _deserialise_meta_data(self, record: t.Dict) -> DataSetMetadata:
        metadata = DataSetMetadata(
            name=record["name"],
            time_level=record["time_level"],
            static=record["static"],
            params=record["params"],
            predecessors={
                pred_record["param_name"]: self._deserialise_metadata_as_stub(
                    pred_record
                )
                for pred_record in record["predecessors"]
            },
            engine=IPersistenceEngine.create_engine(record["engine"]),
        )
        assert metadata.__hash__() == record["hash"]
        return metadata

    def _deserialise_data(self, record: t.Dict, time_range=None):
        data = pickle.loads(record["data"])
        if not record["static"] and time_range is not None:
            data = time_range.view(data, level=record["time_level"])
        return {
            "data": data,
            "declared_time_range": eval(record["declared_time_range"]),
        }

    def _find_record(self, meta_data, include_data=False):
        if self._hash_equality_sufficient:
            return self._find_record_from_hash(
                meta_data.name, meta_data.__hash__(), include_data=include_data
            )
        else:
            raise NotImplementedError

    def _find_record_from_hash(self, name, hash, include_data=False):
        return self._collection.find_one(
            {"name": name, "hash": hash}, None if include_data else {"data": False}
        )

    def get_predecessors_from_hash(
        self, name: str, hash: int
    ) -> t.Dict[str, DatasetMetadataStub]:
        record = self._find_record_from_hash(name, hash, include_data=False)
        return frozendict(
            {
                name: self._deserialise_metadata_as_stub(pred_record)
                for name, pred_record in record["predecessors"].items()
            }
        )

    def exists(self, metadata: DataSetMetadata) -> bool:
        return (
            self._find_record_from_hash(metadata.name, metadata.__hash__()) is not None
        )

    def get_dataset(
        self,
        metadata: DataSetMetadata,
        time_range: t.Optional[TimeRange] = None,
    ) -> DataSet:
        record = self._find_record(metadata, include_data=True)
        if record is not None:
            return DataSet(
                metadata=self._deserialise_meta_data(record),
                **self._deserialise_data(record, time_range)
            )

    def read(
        self, metadata: DataSetMetadata, time_range: t.Optional[TimeRange] = None
    ) -> t.Any:
        return self.get_dataset(metadata, time_range).data

    def get_data_time_range(self, metadata: DataSetMetadata) -> t.Optional[TimeRange]:
        record = self._find_record(metadata, include_data=False)
        return eval(record["data_time_range"])

    def get_declared_time_range(
        self, metadata: DataSetMetadata
    ) -> t.Optional[TimeRange]:
        record = self._find_record(metadata, include_data=False)
        return eval(record["declared_time_range"])

    def idempotent_insert(
        self,
        dataset: DataSet,
    ) -> bool:
        if self.exists(dataset.metadata):
            return True
        else:
            return self.replace(dataset)

    def replace(
        self,
        dataset: DataSet,
    ) -> bool:
        record = self._find_record(dataset.metadata, include_data=False)
        if record is not None:
            self._collection.update_one(
                filter={
                    "name": dataset.metadata.name,
                    "hash": dataset.metadata.__hash__(),
                },
                update={"$set": self._serialise_data(dataset)},
            )
            return True
        else:
            self._collection.insert_one(
                self._serialise_data(dataset)
                | self._serialise_metadata(dataset.metadata)
            )
            return False

    def append(self, dataset):
        existing_dataset = self.get_dataset(dataset.metadata)
        if existing_dataset is None:
            return self.replace(dataset)
        else:
            return self.replace(self._append(existing_dataset, dataset))

    def merge(self, dataset):
        existing_dataset = self.get_dataset(dataset.metadata)
        if existing_dataset is None:
            self.replace(dataset)
        else:
            self.replace(self._merge(existing_dataset, dataset))

    def find_successors(self, metadata: DataSetMetadata) -> t.Set[DataSetMetadata]:
        records = self._collection.find(
            {
                "predecessors.name": metadata.name,
                "predecessors.hash": metadata.__hash__(),
            },
            {"data": False},
        )
        return set((self._deserialise_meta_data(r) for r in records))

    def _delete_leaf(self, metadata: DataSetMetadata):
        if not self.exists(metadata):
            return False
        else:
            successors = self.find_successors(metadata)
            if len(successors) > 0:
                raise ValueError("Cannot delete a dataset that still has successors")
            elif self._hash_equality_sufficient:
                self._collection.delete_one(
                    {"name": metadata.name, "hash": metadata.__hash__()}
                )
                return True
            else:
                raise NotImplementedError

    def delete(self, metadata: DataSetMetadata, recursive=False):
        if recursive:
            for successor in self.find_successors(metadata):
                self.delete(successor, recursive=True)
        return self._delete_leaf(metadata)