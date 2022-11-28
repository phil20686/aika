import pickle
import typing as t
from abc import ABC, abstractmethod

import gridfs
import pymongo
from frozendict._frozendict import frozendict
from overrides import overrides

from aika.datagraph.interface import (
    DataSet,
    DataSetMetadata,
    DataSetMetadataStub,
    IPersistenceEngine,
)
from aika.time.time_range import TimeRange


class IMongoClientCreator(ABC):
    """
    To make the engine pickleable, we must have an entity that can create the clients. Since a new client will
    need to be created on every process or distributed machine.
    """

    @abstractmethod
    def create_client(self) -> pymongo.MongoClient:
        raise NotImplementedError


class UnsecuredLocalhostClient(IMongoClientCreator):
    """
    Creates the default mongo client for connection to a localhost
    """

    def create_client(self) -> pymongo.MongoClient:
        return pymongo.MongoClient()


class MongoBackedPersistanceEngine(IPersistenceEngine):
    """
    This is a mongo bocked version of the persistence engine.

    Notes
    -----
    [1] Throughout we use metadata.__hash__() rather than hash(metadata) since we
    will look in the future to make our hashes extremely unique, and hash() truncates
    the output of __hash__().

    [2] This backend has not been tested with concurrent read/writes to the same datasets.
    It is likely to fail in that situation. This is relatively unavoidable since gridfs for mongo db
    does not support transactions as of writing. So there is no way to insure atomic file updates. However,
    it is the nature of the completion checking logic of the tasks that a dataset should only ever be being
    written by a single task, and that nothing should attempt to read it until that is marked complete, so
    this need not block normal envisaged use.
    """

    def _init_derived_properties(self):
        self._client = self._client_creator.create_client()
        self._database = self._client.get_database(self._database_name)
        self._collection = self._database.get_collection(self._collection_name)
        self._gridfs = gridfs.GridFS(
            self._database, collection=self._collection_name + "_grid_fs"
        )

    def __init__(
        self,
        client_creator: IMongoClientCreator,
        database_name="datagraph",
        collection_name="default",
    ):
        self._client_creator = client_creator
        self._database_name = database_name
        self._collection_name = collection_name
        self._serialise_mode = "pickle"
        self._hash_equality_sufficient = True
        self._init_derived_properties()

    def __hash__(self):
        return hash((self._database_name, self._serialise_mode))

    def __getstate__(self):
        return {
            "client_creator": self._client_creator,
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "serialise_mode": self._serialise_mode,
            "hash_equality_sufficient": self._hash_equality_sufficient,
        }

    def __setstate__(self, state):
        self._client_creator = state["client_creator"]
        self._database_name = state["database_name"]
        self._collection_name = state["collection_name"]
        self._serialise_mode = state["serialise_mode"]
        self._hash_equality_sufficient = state["hash_equality_sufficient"]
        self._init_derived_properties()

    def __eq__(self, other):
        if isinstance(other, MongoBackedPersistanceEngine):
            return all(
                (
                    self._database_name == other._database_name,
                    self._serialise_mode == other._serialise_mode,
                )
            )
        else:
            return NotImplemented  # pragma: no cover

    @classmethod
    def _create_engine(cls, d: t.Dict[str, t.Any]) -> "MongoBackedPersistanceEngine":
        client_creator = pickle.loads(d.pop("client_creator"))
        return cls(client_creator=client_creator, **d)

    @overrides()
    def set_state(self) -> t.Dict[str, t.Any]:
        return {
            "type": "mongodb",
            "client_creator": pickle.dumps(self._client_creator),
            "database_name": self._database_name,
            "collection_name": self._collection_name,
        }

    def _serialise_metadata_as_stub(self, metadata: DataSetMetadata):
        return {
            "name": metadata.name,
            "hash": metadata.__hash__(),
            "time_level": metadata.time_level,
            "static": metadata.static,
            "version": metadata.version,
            "params": metadata.params,
            "engine": metadata.engine.set_state(),
        }

    def _serialise_metadata(self, metadata: DataSetMetadata):
        return {
            "name": metadata.name,
            "hash": metadata.__hash__(),
            "time_level": metadata.time_level,
            "static": metadata.static,
            # TODO : remove cast when issue https://github.com/mongomock/mongomock/issues/814 is resolved.
            "params": dict(metadata.params),
            "version": metadata.version,
            "engine": metadata.engine.set_state(),
            "predecessors": [
                self._serialise_metadata_as_stub(pred) | {"param_name": name}
                for name, pred in metadata.predecessors.items()
            ],
        }

    def _serialise_data_metadata(self, dataset: DataSet):
        return {
            # "data": pickle.dumps(dataset.data),
            "declared_time_range": repr(dataset.declared_time_range),
            "data_time_range": repr(dataset.data_time_range),
        }

    def _deserialise_metadata_as_stub(self, record: t.Dict):
        return DataSetMetadataStub(
            name=record["name"],
            static=record["static"],
            params=record["params"],
            version=record["version"],
            hash=record["hash"],
            time_level=record["time_level"],
            engine=IPersistenceEngine.create_engine(record["engine"]),
        )

    def _deserialise_meta_data(self, record: t.Dict) -> DataSetMetadata:
        metadata = DataSetMetadata(
            name=record["name"],
            time_level=record["time_level"],
            static=record["static"],
            params=record["params"],
            version=record["version"],
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

    def _deserialise_data_metadata(self, record: t.Mapping, time_range=None):
        data = pickle.loads(record["data"])
        if not record["static"] and time_range is not None:
            data = time_range.view(data, level=record["time_level"])
        return {
            "data": data,
            "declared_time_range": eval(record["declared_time_range"]),
        }

    def _find_record(self, meta_data: DataSetMetadata, include_data=False):
        if self._hash_equality_sufficient:
            return self._find_record_from_hash(
                meta_data.name, meta_data.__hash__(), include_data=include_data
            )
        else:
            raise NotImplementedError  # pragma: no cover

    def _find_record_from_hash(self, name, hash, include_data=False):
        record = self._collection.find_one({"name": name, "hash": hash})
        if include_data and record is not None:
            record["data"] = self._gridfs.get(file_id=record["_id"]).read(
                size=-1
            )  # read it all
        return record

    @overrides
    def get_predecessors_from_hash(
        self, name: str, hash: int
    ) -> t.Dict[str, DataSetMetadataStub]:
        record = self._find_record_from_hash(name, hash, include_data=False)

        if record is not None:
            return frozendict(
                {
                    pred_record["param_name"]: self._deserialise_metadata_as_stub(
                        pred_record
                    )
                    for pred_record in record["predecessors"]
                }
            )
        else:
            raise ValueError(f"No datasets {name} {hash}")

    @overrides()
    def exists(self, metadata: DataSetMetadata) -> bool:
        return (
            self._find_record_from_hash(metadata.name, metadata.__hash__()) is not None
        )

    @overrides()
    def get_dataset(
        self,
        metadata: DataSetMetadata,
        time_range: t.Optional[TimeRange] = None,
    ) -> DataSet:
        if metadata.static and time_range is not None:
            raise ValueError("time_range must be None for static datasets")
        record = self._find_record(metadata, include_data=True)
        if record is not None:
            data = self._gridfs.get(record["_id"])
            return DataSet(
                metadata=self._deserialise_meta_data(record),
                **self._deserialise_data_metadata(record, time_range),
            )

    @overrides()
    def read(
        self, metadata: DataSetMetadata, time_range: t.Optional[TimeRange] = None
    ) -> t.Any:
        dataset = self.get_dataset(metadata, time_range)
        if dataset is None:
            return None
        else:
            return dataset.data

    @overrides()
    def get_data_time_range(self, metadata: DataSetMetadata) -> t.Optional[TimeRange]:
        if metadata.static:
            raise ValueError("No declared time range for static data")
        else:
            record = self._find_record(metadata, include_data=False)
            if record is not None:
                return eval(record["data_time_range"])

    @overrides()
    def get_declared_time_range(
        self, metadata: DataSetMetadata
    ) -> t.Optional[TimeRange]:
        if metadata.static:
            raise ValueError("No declared time range for static data")
        else:
            record = self._find_record(metadata, include_data=False)
            if record is not None:
                return eval(record["declared_time_range"])

    @overrides()
    def idempotent_insert(
        self,
        dataset: DataSet,
    ) -> bool:
        if self.exists(dataset.metadata):
            return True
        else:
            return self.replace(dataset)

    @overrides()
    def replace(
        self,
        dataset: DataSet,
    ) -> bool:
        record = self._find_record(dataset.metadata, include_data=False)
        if record is not None:
            self._gridfs.delete(record["_id"])
            self._gridfs.put(
                data=pickle.dumps(dataset.data),
                _id=record["_id"],
            )
            self._collection.update_one(
                filter={
                    "name": dataset.metadata.name,
                    "hash": dataset.metadata.__hash__(),
                },
                update={"$set": self._serialise_data_metadata(dataset)},
            )
            return True
        else:

            foo = self._collection.insert_one(
                self._serialise_data_metadata(dataset)
                | self._serialise_metadata(dataset.metadata)
            )
            self._gridfs.put(data=pickle.dumps(dataset.data), _id=foo.inserted_id)
            return False

    @overrides()
    def append(self, dataset) -> bool:
        if dataset.metadata.static:
            raise ValueError("Can only append for time-series data")
        existing_dataset = self.get_dataset(dataset.metadata)
        if existing_dataset is None:
            return self.replace(dataset)
        else:
            return self.replace(self._append(existing_dataset, dataset))

    @overrides()
    def merge(self, dataset) -> bool:
        if dataset.metadata.static:
            raise ValueError("Can only merge for time-series data")
        existing_dataset = self.get_dataset(dataset.metadata)
        if existing_dataset is None:
            self.replace(dataset)
        else:
            self.replace(self._merge(existing_dataset, dataset))

    @overrides()
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
                record = self._find_record(metadata, include_data=False)
                self._collection.delete_one(
                    {"name": metadata.name, "hash": metadata.__hash__()}
                )
                self._gridfs.delete(record["_id"])
                return True
            else:
                raise NotImplementedError  # pragma: no cover

    @overrides()
    def delete(self, metadata: DataSetMetadata, recursive=False):
        if recursive:
            for successor in self.find_successors(metadata):
                self.delete(successor, recursive=True)
        return self._delete_leaf(metadata)

    @overrides()
    def find(self, match: str, version: t.Optional[str] = None) -> t.List[str]:
        search_terms = {"name": {"$regex": match}}
        if version is not None:
            search_terms["version"] = version
        return list(
            sorted(
                [
                    record["name"]
                    for record in self._collection.find(search_terms, {"name": True})
                ]
            )
        )

    @overrides()
    def scan(
        self, dataset_name: str, params: t.Optional[t.Dict] = None
    ) -> t.Set[DataSetMetadataStub]:
        search_terms = {"name": dataset_name}
        candidates = {
            self._deserialise_metadata_as_stub(record)
            for record in self._collection.find(search_terms)
        }
        if params:
            results = set()
            for candidate in candidates:

                if all(
                    [
                        candidate.recursively_get_parameter_value(param) == value
                        for param, value in params.items()
                    ]
                ):
                    results.add(candidate)
            return results
        else:
            return candidates
