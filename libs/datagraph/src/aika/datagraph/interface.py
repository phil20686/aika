import typing as t
from abc import ABC, abstractmethod

import attr
import pandas as pd
from frozendict import frozendict

from aika.time.time_range import TimeRange
from aika.utilities.pandas_utils import IndexTensor, equals


class DataSetMetadata:
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

        return type(self)(
            name=self.name,
            static=self.static,
            params=self.params,
            predecessors=predecessors,
            time_level=self.time_level,
            engine=engine,
        )

    def __init__(
        self,
        *,
        name: str,
        static: bool,
        params: t.Dict[str, t.Any],
        predecessors: t.Dict[str, "DataSetMetadata"],
        time_level: t.Optional[t.Union[int, str]] = None,
        engine: t.Optional["iPersitenceEngine"] = None,
    ):
        self._name = name
        self._static = static
        self._engine = engine
        self._time_level = time_level
        self._params = frozendict({k: params[k] for k in sorted(params)})
        self._predecessors = frozendict(
            {k: predecessors[k] for k in sorted(predecessors)}
        )

    def __eq__(self, other):
        return all(
            (
                self._name == other.name,
                self._static == other.static,
                self._engine == other.engine,
                self._time_level == other.time_level,
                self._params == other.params,
                self.__hash__() == other.__hash__(),
            )
        )

    def __hash__(self):
        return hash(
            (
                self._name,
                self._static,
                self._time_level,
                self._params,
            )
            + tuple(hash(x) for x in self._predecessors.values())
        )

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
    def engine(self) -> "iPersistenceEngine":
        return self._engine

    @property
    def params(self):
        return self._params

    @property
    def predecessors(self) -> t.Dict[str, "DataSetMetadata"]:
        return self._predecessors

    def is_immediate_predecessor(self, metadata: "DataSetMetadata") -> bool:
        """
        Returns true if the given metadata represents one of the immediate predecessors.
        """
        return metadata in set(self._predecessors.values())

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


class DatasetMetadataStub(DataSetMetadata):
    """
    A stub class is different only because it stores only the hash and engine + top level parameters
    directly, and will fetch the full predecessors when required.
    """

    def __init__(
        self,
        *,
        name: str,
        static: bool,
        params: t.Dict[str, t.Any],
        hash: int,
        time_level: t.Optional[t.Union[int, str]] = None,
        engine: t.Optional["iPersitenceEngine"] = None,
    ):
        self._name = name
        self._static = static
        self._engine = engine
        self._time_level = time_level
        self._params = frozendict({k: params[k] for k in sorted(params)})
        self._hash = hash

    def __hash__(self):
        return self._hash

    @property
    def predecessors(self) -> t.Dict[str, "DatasetMetadataStub"]:
        return self.engine.get_predecessors_from_hash(self._name, self._hash)


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
                raise ValueError(f"{attribute} must be None for static data")
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

    # ----------------------------------------------------------------------------------
    # Read-only methods
    # ----------------------------------------------------------------------------------

    @abstractmethod
    def set_state(self) -> t.Dict[str, t.Any]:
        raise NotImplementedError

    @classmethod
    def create_engine(cls, d: t.Dict[str, t.Any]):
        d = (
            d.copy()
        )  # should not alter the dict, which may be a database record of some type.
        engine_type = d.pop("type")
        if engine_type == "hash_backed":
            raise NotImplementedError("Cannot recreate an engine for in-memory storage")
        elif engine_type == "mongodb":
            from aika.datagraph.persistence.mongo_backed import (
                MongoBackedPersistanceEngine,
            )

            return MongoBackedPersistanceEngine._create_engine(d)
        else:
            raise NotImplementedError(f"No persistence engine found for {d['type']}")

    @abstractmethod
    def get_predecessors_from_hash(
        self, name: str, hash: int
    ) -> t.Dict[str, DatasetMetadataStub]:
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
        metadata : DatasetMetadata

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
        metadata : DatasetMetadata

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
    def find_successors(self, metadata: DataSetMetadata) -> t.Set[DatasetMetadataStub]:
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
