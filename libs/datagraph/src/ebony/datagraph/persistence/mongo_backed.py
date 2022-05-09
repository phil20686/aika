import pymongo
from bson.objectid import ObjectId
from ebony.datagraph.interface import IPersistenceEngine, DataSetMetadata
import pickle
import typing as t

from ebony.time.time_range import TimeRange


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
            client : pymongo.MongoClient,
            database_name="datagraph"
    ):
        self._client = client
        self._database_name = database_name
        self._database = self._client.get_database(database_name)
        self._serialise_mode = "pickle"
        self._hash_equality_sufficient = True

    def __hash__(self):
        return hash((self._database_name, self._serialise_mode))

    def _serialise_meta_data(self, metadata: DataSetMetadata):
        return {
            "name": metadata.name,
            "hash": metadata.__hash__(),
            "time_level": metadata.time_level,
            "static": metadata.static,
            "params": metadata.params,
            "predecessors": {
                name:self._serialise_meta_data(pred) for name,pred in metadata.predecessors.items()}
        }

    def _deserialise_meta_data(self, record: t.Dict) -> DataSetMetadata:
        metadata = DataSetMetadata(
            name=record["name"],
            time_level=record["time_level"],
            static=record["static"],
            params=record["params"],
            predecessors={name:self._deserialise_meta_data(pred) for name,pred in record["predecessors"].items()},
            engine=self
        )
        assert metadata.__hash__() == record["hash"]
        return metadata

    def _deserialise_data(self, record: t.Dict):
        return {
            "data" : pickle.loads(record["data"]),
            "declared_time_range": eval(record["declared_time_range"]),
            "data_time_range": eval(record["data_time_range"])
        }


    def exists(self, metadata: DataSetMetadata) -> bool:
        if self._hash_equality_sufficient:
            return self._database.get_collection(metadata.name).find_one(
                filter=({"hash": metadata.__hash__()})
            ) is not None
        else:
            raise NotImplementedError

    def get_dataset(
        self,
        metadata: DataSetMetadata,
        time_range: t.Optional[TimeRange] = None,
    ) -> DataSet:
        if self._hash_equality_sufficient:
            record = self._database.get_collection(metadata.name).find_one(
                filter=({"hash": metadata.__hash__()})
            )

        else:
            raise NotImplementedError







