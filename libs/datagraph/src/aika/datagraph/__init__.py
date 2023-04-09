from ._version import version
from .interface import DataSetMetadataStub, DataSetMetadata, DataSet, IPersistenceEngine
from .persistence.mongo_backed import MongoBackedPersistanceEngine
from .persistence.hash_backed import HashBackedPersistanceEngine
