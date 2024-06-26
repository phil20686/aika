import pickle
import typing as t
from abc import ABC, abstractmethod

import attr
import pandas as pd
from frozendict import frozendict
from overrides import overrides

from aika.datagraph.utils import normalize_parameters
from aika.time import TimeRange
from aika.time.time_range import TimeRange
from aika.utilities.freezing import unfreeze_recursively
from aika.utilities.hashing import session_consistent_hash
from aika.utilities.pandas_utils import IndexTensor, equals


class DataSetMetadataStub:
    """
    A stub class is different only because it stores only the hash and engine + top level parameters
    directly, and will fetch the full predecessors when required.
    """

    def replace_engine(self, engine):
        """
        Useful for testing, not for production code.
        """

        return DataSetMetadataStub(
            name=self.name,
            static=self.static,
            params=self.params,
            version=self.version,
            time_level=self.time_level,
            engine=engine,
            hash=self._hash,
        )

    def __init__(
        self,
        *,
        name: str,
        static: bool,
        params: t.Dict[str, t.Any],
        hash: int,
        version: str,
        time_level: t.Optional[t.Union[int, str]] = None,
        engine: t.Optional["IPersistenceEngine"] = None,
    ):
        self._name = name
        self._static = static
        self._engine = engine
        self._version = version
        self._time_level = time_level
        # self._params = frozendict({k: params[k] for k in sorted(params)})
        self._params = normalize_parameters(params)
        self._hash = hash

    def __eq__(self, other):
        return all(
            (
                self._name == other.name,
                self._static == other.static,
                self._engine == other.engine,
                self._version == other._version,
                self._time_level == other.time_level,
                self._params == other.params,
                self.__hash__() == other.__hash__(),
            )
        )

    def __hash__(self):
        return self._hash

    @property
    def name(self) -> str:
        return self._name

    @property
    def static(self) -> bool:
        return self._static

    @property
    def time_level(self) -> t.Union[str, int]:
        return self._time_level

    @property
    def version(self):
        return self._version

    @property
    def engine(self) -> "IPersistenceEngine":
        return self._engine

    @property
    def params(self) -> frozendict:
        return self._params

    @property
    def predecessors(self) -> t.Dict[str, "DataSetMetadataStub"]:
        return self.engine.get_predecessors_from_hash(
            self._name, self._version, self._hash
        )

    def exists(self):
        return self.engine.exists(self)

    def get_dataset(self, time_range: t.Optional[TimeRange] = None) -> "DataSet":
        return self.engine.get_dataset(metadata=self, time_range=time_range)

    def read(self, time_range: t.Optional[TimeRange] = None) -> t.Any:
        return self.engine.read(metadata=self, time_range=time_range)

    def get_data_time_range(self) -> t.Optional[TimeRange]:
        return self.engine.get_data_time_range(self)

    def get_declared_time_range(self) -> t.Optional[TimeRange]:
        return self.engine.get_declared_time_range(self)

    def idempotent_insert(
        self,
        declared_time_range: t.Optional[TimeRange],
        data: t.Any,
    ) -> bool:
        return self.engine.idempotent_insert(
            DataSet(
                metadata=self,
                declared_time_range=declared_time_range,
                data=data,
            )
        )

    def replace(
        self,
        declared_time_range: t.Optional[TimeRange],
        data: t.Any,
    ) -> bool:
        return self.engine.replace(
            DataSet(
                metadata=self,
                declared_time_range=declared_time_range,
                data=data,
            )
        )

    def append(
        self,
        declared_time_range: t.Optional[TimeRange],
        data: t.Any,
    ) -> bool:
        return self.engine.append(
            DataSet(
                metadata=self,
                declared_time_range=declared_time_range,
                data=data,
            )
        )

    def merge(
        self,
        declared_time_range: t.Optional[TimeRange],
        data: t.Any,
    ) -> bool:
        return self.engine.merge(
            DataSet(
                metadata=self,
                declared_time_range=declared_time_range,
                data=data,
            )
        )

    def is_immediate_predecessor(self, metadata: "DataSetMetadata") -> bool:
        """
        Returns true if the given metadata represents one of the immediate predecessors.
        """
        return metadata in set(self.predecessors.values())

    def get_parameter_value(self, param_name):
        if param_name == "version":
            return self.version
        elif param_name == "time_level":
            return self.time_level
        elif param_name == "static":
            return self.static
        elif param_name == "name":
            return self.name
        else:
            try:
                return self._params[param_name]
            except KeyError:
                raise ValueError(f"Dataset {self.name} has no parameter {param_name}")

    def recursively_get_parameter_value(self, param_name):
        """
        If param name contains dots, it is assumed that it refers to a dataset by a predecessors value,
        so foo.bar means get parameter value bar from predecessor foo. This allows one to conveniently
        `walk the graph` when getting a dataset that differs only by the parameterisation of a parent.

        Parameters
        ----------
        param_name: str

        Returns
        -------
        Any: The value of the parameter
        """
        target_metadata = self
        path = param_name.split(".")
        for predecessor in path[:-1]:
            try:
                target_metadata = target_metadata.predecessors[predecessor]
            except KeyError:
                raise ValueError(
                    f"Dataset {target_metadata.name} had no predecessor under the name {predecessor}"
                )
        return target_metadata.get_parameter_value(path[-1])


class DataSetMetadata(DataSetMetadataStub):
    """
    A `DataSetMetadata` object contains all the information to describe a dataset which
    may or may not exist. Note that dataset metadata equality is explicitly based on the hashes of predecessors.
    hash() truncates the output of __hash__() so we try to use __hash__() throughout when eg writing meta data
    to a database. Note that metadata.
    """

    def replace_engine(self, engine, include_predecessors=False):
        """
        Useful for testing, not for production code.

        """
        if include_predecessors:
            predecessors = {
                name: m.replace_engine(engine, include_predecessors)
                for name, m in self.predecessors.items()
            }
        else:
            predecessors = self.predecessors

        return DataSetMetadata(
            name=self.name,
            static=self.static,
            params=self.params,
            version=self.version,
            predecessors=predecessors,
            time_level=self.time_level,
            engine=engine,
        )

    def __init__(
        self,
        *,
        name: str,
        static: bool,
        version: str,
        params: t.Dict[str, t.Any],
        predecessors: t.Dict[str, "DataSetMetadata"],
        time_level: t.Optional[t.Union[int, str]] = None,
        engine: t.Optional["IPersistenceEngine"] = None,
    ):
        self._name = name
        self._static = static
        self._engine = engine
        self._version = version
        if static and time_level is not None:
            raise ValueError("Cannot specify a time level on static data")
        self._time_level = time_level
        self._params = normalize_parameters(params)
        self._predecessors = frozendict(
            {k: predecessors[k] for k in sorted(predecessors)}
        )
        self.__hash = None

    def __hash__(self):
        if self.__hash is None:
            self.__hash = session_consistent_hash(
                (
                    self._name,
                    self._static,
                    self._time_level,
                    self._version,
                    self._engine,
                    self._params,
                )
                + tuple(hash(x) for x in self._predecessors.values())
            )
        return self.__hash

    @property
    def predecessors(self) -> t.Dict[str, "DataSetMetadata"]:
        return self._predecessors


# TODO: split this into StaticDataSet and TimeSeriesDataSet?
# TODO: does this need to be hashable? It will currently fail because `data` will
#  usually be an unhashable pandas object
@attr.s(frozen=True, eq=True, hash=False)
class DataSet:
    @classmethod
    def build(
        cls,
        name: str,
        data: IndexTensor,
        params: t.Dict,
        predecessors: t.Dict,
        version: str = "no_version",
        static: bool = False,
        time_level: t.Optional[t.Union[str, int]] = None,
        engine=None,
        declared_time_range: t.Optional[TimeRange] = None,
    ):
        return cls(
            metadata=DataSetMetadata(
                name=name,
                params=params,
                predecessors=predecessors,
                static=static,
                version=version,
                time_level=time_level,
                engine=engine,
            ),
            data=data,
            declared_time_range=declared_time_range
            or (None if static else TimeRange.from_pandas(data, level=time_level)),
        )

    def replace_engine(self, engine, include_predecessors=False):
        """
        Useful for testing, allows test cases to be reused for testing different engines.
        """
        return type(self)(
            data=self.data,
            metadata=DataSetMetadata.replace_engine(
                self.metadata, engine, include_predecessors
            ),
            declared_time_range=self.declared_time_range,
        )

    metadata: DataSetMetadata = attr.ib()
    # TODO: Remove below PyCharm exception once attrs 22.1 is released, fixing the bug
    #  mentioned here: https://github.com/python-attrs/attrs/issues/948
    # noinspection PyUnresolvedReferences
    data: t.Union[IndexTensor, t.Any] = attr.ib(eq=attr.cmp_using(equals))
    declared_time_range: t.Optional[TimeRange] = attr.ib()

    @declared_time_range.validator
    def validate_declared_time_range(self, attribute, value):
        if self.metadata.static:
            if value is not None:
                raise ValueError(f"{attribute.name} must be None for static data")
        elif value is None:
            raise ValueError("Must declare a time range for non-static dataset.")
        else:
            if not value.contains(self.data_time_range):
                raise ValueError(
                    f"Invalid declared_time_range {value}. Must be a"
                    f"superset of the actual data's time range {self.data_time_range}"
                )

    @property
    def data_time_range(self):
        """
        Note that the data time range always applies to the data contained in this dataset object, which
        may only be a subset of the data that is stored in the persistence engine.
        """
        if self.metadata.static:
            return None
        else:
            return TimeRange.from_pandas(self.data, level=self.metadata.time_level)

    def update(self, data, declared_time_range):
        """
        Creates a new dataset that has the same parameters
        but different data
        """
        return DataSet(
            metadata=self.metadata,
            data=data,
            declared_time_range=declared_time_range,
        )


class IPersistenceEngine(ABC):

    """
    The model of our persistence engines here is that "the data is the state",
    so you can have lots of persistence engines pointing at a database and
    they should be functionally identical. Ideally we would make sure that
    all operations are fully atomic, but in practice this is the back end
    for a graph computation, and graph runners will make sure that the tasks
    run in the right order.
    """

    # ----------------------------------------------------------------------------------
    # Read-only methods
    # ----------------------------------------------------------------------------------

    @abstractmethod
    def set_state(self) -> t.Dict[str, t.Any]:
        """
        This method will return a dict that represents the information
        that is needed to allow a engine to be recreated. It must pair
        with the create implementation so that
        IPersistence.create_engine(**engine_instance.set_state())
        will create a functionally identical persistence engine.

        Returns
        -------

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def create_engine(cls, d: t.Dict[str, t.Any]):
        """
        It is possible to create datasets that point to datasets in a different
        database owned by a different engine, to do this, we must be able to create
        persistence engines out of data stored along with the predecessors. That is
        what this classmethod does, it is used inside persistent storage.
        """
        d = (
            d.copy()
        )  # should not alter the dict, which may be a database record of some type.
        engine_type = d.pop("type")
        if engine_type == "hash_backed":
            raise NotImplementedError(
                "Cannot recreate an engine for in-memory storage"
            )  # pragma: no cover
        elif engine_type == "mongodb":
            from aika.datagraph.persistence.mongo_backed import (
                MongoBackedPersistanceEngine,
            )

            return MongoBackedPersistanceEngine._create_engine(d)
        elif engine_type == "pure_filesystem":
            from aika.datagraph.persistence.pure_filesystem_backend import (
                FileSystemPersistenceEngine,
            )

            return FileSystemPersistenceEngine(**d)
        else:
            raise NotImplementedError(
                f"No persistence engine found for {d['type']}"
            )  # pragma: no cover

    @abstractmethod
    def get_predecessors_from_hash(
        self, name: str, version: str, hash: int
    ) -> t.Mapping[str, DataSetMetadataStub]:
        """
        Given a node name and the hash of a dataset, returns a dataset stub.
        """

    @abstractmethod
    def exists(self, metadata: DataSetMetadata) -> bool:
        """
        Return a boolean indicating whether the dataset identified by `metadata` exists.
        """

    @abstractmethod
    def get_dataset(
        self,
        metadata: DataSetMetadata,
        time_range: t.Optional[TimeRange] = None,
    ) -> DataSet:
        """
        Return the dataset identified by `metadata`, if it exists.

        If `time_range` is specified, restrict the data to `time_range.view(data)`

        Parameters
        ----------
        metadata : DataSetMetadata

            The metadata identifying which dataset to fetch

        time_range : TimeRange or None, default None

            The time range to slice the data to.
            If None, all the data is returned without t.Any slicing.

        Returns
        -------

        DataSet or None

            The specified dataset, or None if it does not exist

        Raises
        ------

        ValueError

            If `time_range` is not None but the specified dataset is static.
        """

    @abstractmethod
    def read(
        self, metadata: DataSetMetadata, time_range: t.Optional[TimeRange] = None
    ) -> t.Any:
        """
        Return the data for the dataset identified by `metadata`, if it exists.

        If `time_range` is specified, return only the slice of the data contained
        within `time_range`.

        Should be equivalent to

        ```
        self.get_dataset(metadata=metadata, time_range=time_range).data
        ```

        But may be more efficient, depending on the engine implementation.

        Parameters
        ----------
        metadata : DataSetMetadata

            The metadata identifying which dataset to fetch

        time_range : TimeRange or None, default None

            The time range to slice the data to.
            If None, all the data is returned without t.Any slicing.

        Returns
        -------

        t.Any

            The data matching the specified dataset, or None if it does not exist

        Raises
        ------

        ValueError

            If `time_range` is not None but the specified dataset is static.
        """

    @abstractmethod
    def get_data_time_range(self, metadata: DataSetMetadata) -> t.Optional[TimeRange]:
        """
        Get the time range covered by the data of the dataset identified by `metadata`.

        Should be equivalent to

        ```
            TimeRange.from_pandas(self.read(metadata))
        ```

        But may be more efficient, depending on the engine implementation.

        Parameters
        ----------
        metadata : DataSetMetadata

            The metadata identifying the dataset to be queried

        Returns
        -------
        TimeRange or None

            The time range covered by the data of the dataset identified by `metadata`,
            if it exists; None otherwise

        Raises
        ------
        ValueError

            If the specified dataset is static
        """

    @abstractmethod
    def get_declared_time_range(
        self, metadata: DataSetMetadata
    ) -> t.Optional[TimeRange]:
        """
        Get the declared time range of the dataset identified by `metadata`.

        Should be equivalent to

        ```
        self.get_dataset(metadata=metadata, time_range=time_range).declared_time_range
        ```

        But may be more efficient, depending on the engine implementation.

        Parameters
        ----------
        metadata : DataSetMetadata

            The metadata identifying the dataset to be queried

        Returns
        -------
        TimeRange or None

            The declared time range of the dataset identified by `metadata`, if it
            exists; None otherwise

        Raises
        ------
        ValueError

            If the specified dataset is static
        """

    # ----------------------------------------------------------------------------------
    # Write methods
    # ----------------------------------------------------------------------------------

    @abstractmethod
    def idempotent_insert(
        self,
        dataset: DataSet,
    ) -> bool:
        """
        Persist a new dataset.

        If there is an existing dataset with the same metadata, do nothing, leaving
        the existing dataset unchanged.

        Returns true if a dataset already existed.
        """

    @abstractmethod
    def replace(
        self,
        dataset: DataSet,
    ) -> bool:
        """
        Persist a new dataset.

        If there is an existing dataset with the same metadata, it will be replaced.

        Returns true if a dataset already existed.
        """

    @abstractmethod
    def append(
        self,
        dataset: DataSet,
    ) -> bool:
        """
        Persist a new dataset.

        If there is an existing dataset `ex`, then it will be updated to a combined
        dataset `c`, defined by appending the new rows of `new.data` to `ex.data`. In
        detail, the semantics should be as follows:

        - `c.data = pd.concat([ex.data, new_time_range.view(new.data)])`, where
           `new_time_range := TimeRange(ex.data_time_range.end, None)`

        - ```c.declared_time_range = TimeRange(
                 ex.declared_time_range.start,
                 max(ex.declared_time_range.end, new.declared_time_range.end))```

        If †here is no existing dataset, `new` will be persisted exactly as-is.

        Returns true iff a dataset already existed.

        Note that this method is only supported for time-series datasets; if the
        specified dataset is static, a ValueError will be raised.
        """

    @abstractmethod
    def merge(
        self,
        dataset: DataSet,
    ) -> bool:
        """
        Persist a new dataset.

        If there is an existing dataset `ex`, then it will be updated to a combined
        dataset `c`, defined by merging `c.data` with `new.data`.

        In detail, the semantics should be as follows:

        - `c.data = ex.data.combine_first(new.data)` if `data` is a Series or DataFrame,
          and `c.data = ex.data.union(new.data)` if `data` is an Index.

        - ```c.declared_time_range = TimeRange(
                 min(ex.declared_time_range.start, new.declared_time_range.start),
                 max(ex.declared_time_range.end, new.declared_time_range.end))```

        If †here is no existing dataset, `new` will be persisted exactly as-is.

        Returns true iff a dataset already existed.

        Note that this method is only supported for time-series datasets; if the
        specified dataset is static, a ValueError will be raised.
        """

    def _append(self, existing: DataSet, new: DataSet):
        """
        This is the definitionally correct logic for appending two datasets, all implementations of append
        by different engines must replicate this behaviour.
        """
        new_data = TimeRange(existing.data_time_range.end, None).view(
            new.data, level=new.metadata.time_level
        )

        if new_data.empty:
            return existing

        return DataSet(
            metadata=new.metadata,
            data=pd.concat([existing.data, new_data], axis=0),
            declared_time_range=TimeRange(
                existing.declared_time_range.start,
                # it might merge nothing if the new time range is < existing.
                max(new.declared_time_range.end, existing.declared_time_range.end),
            ),
        )

    def _merge(self, existing: DataSet, new: DataSet) -> DataSet:
        """
        This is the definitionally correct logic for merging two datasets, all implementations of merge
        by different engines must replicate this behaviour.
        """
        return DataSet(
            metadata=new.metadata,
            data=existing.data.combine_first(new.data),
            declared_time_range=existing.declared_time_range.union(
                new.declared_time_range
            ),
        )

    @abstractmethod
    def delete(self, metadata: DataSetMetadata, recursive=False):
        """
        This will delete the dataset, if it has successors then it will fail if recursive is false, or it will
        delete all the children if possible.

        Parameters
        ----------
        metadata : DataSetMetadata
            The metadata of the dataset to delete.
        recursive : bool
            Whether to delete all successors of this dataset.

        Returns
        -------
        bool : Whether any dataset was deleted
        """

    @abstractmethod
    def find_successors(self, metadata: DataSetMetadata) -> t.Set[DataSetMetadataStub]:
        """
        This method will find any successor datasets of a datasets.

        Parameters
        ----------
        metadata : DataSetMetadata
            The dataset to find the successors of.

        Returns
        -------
        set : A list of dataset metadata stubs representing the children at depth one. There may be further
        children.
        """

    @abstractmethod
    def find(self, match: str, version: t.Optional[str] = None) -> t.List[str]:
        """
        This method will find all datasets where the name matches the regex pattern
        given in match. Optionally ignore names which have no dataset with the given version.

        Parameters
        ----------
        match : str
            A regex patter passed directly to re.match
        version : str
            A version string to restrict the results

        Returns
        -------
        List[str] :  the names of the data-nodes in this engine in alphabetical order.
        """

    @abstractmethod
    def scan(
        self, dataset_name: str, params: t.Optional[t.Dict] = None
    ) -> t.Set[DataSetMetadataStub]:
        """
        Given a dataset name, and a set of parameters, will find each dataset in that name which
        matches the params given. If no params are given will return all datasets under that name.
        Note that you can specify queries on upstream parameters, so "foo.baz: 4.0" means scan for
        datasets where predecessor foo's parameter baz has value 4.0. This is helpful because
        often datasets differ only by the value of an upstream parameter.

        Parameters
        ----------
        dataset_name: str
        params: t.Optional[t.Dict]
            A set of parameters for the dataset.

        Returns
        -------
        t.Set[DataSetMetadataStub] : A list of all datasets that met the search criteria.
        """


class _SerialisingBase(IPersistenceEngine):
    def _serialise_metadata_as_stub(self, metadata: DataSetMetadata):
        return {
            "name": metadata.name,
            "hash": metadata.__hash__(),
            "time_level": metadata.time_level,
            "static": metadata.static,
            "version": metadata.version,
            "params": unfreeze_recursively(metadata.params),
            "engine": metadata.engine.set_state(),
        }

    def _serialise_metadata(self, metadata: DataSetMetadata):
        return {
            "name": metadata.name,
            "hash": metadata.__hash__(),
            "time_level": metadata.time_level,
            "static": metadata.static,
            # TODO : remove cast when issue https://github.com/mongomock/mongomock/issues/814 is resolved.
            "params": unfreeze_recursively(metadata.params),
            "version": metadata.version,
            "engine": metadata.engine.set_state(),
            "predecessors": [
                {**self._serialise_metadata_as_stub(pred), **{"param_name": name}}
                for name, pred in metadata.predecessors.items()
            ],
        }

    def _serialise_data_metadata(self, dataset: DataSet):
        return {
            "declared_time_range": repr(dataset.declared_time_range),
            "data_time_range": repr(dataset.data_time_range),
        }

    def _make_record(self, dataset: DataSet):
        return {
            **self._serialise_metadata(dataset.metadata),
            **self._serialise_data_metadata(dataset),
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

    @abstractmethod
    def _find_record(self, metadata: DataSetMetadata, include_data=False):
        raise NotImplementedError

    @abstractmethod
    def _find_record_from_hash(self, name, version, hash, include_data=False):
        raise NotImplementedError

    @overrides
    def get_predecessors_from_hash(
        self, name: str, version: str, hash: int
    ) -> t.Dict[str, DataSetMetadataStub]:
        record = self._find_record_from_hash(name, version, hash, include_data=False)
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
            raise ValueError(f"No datasets {name} {version} {hash}")

    @overrides()
    def exists(self, metadata: DataSetMetadata) -> bool:
        return (
            self._find_record_from_hash(
                metadata.name, metadata.version, metadata.__hash__()
            )
            is not None
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
    def idempotent_insert(
        self,
        dataset: DataSet,
    ) -> bool:
        if self.exists(dataset.metadata):
            return True
        else:
            return self.replace(dataset)
