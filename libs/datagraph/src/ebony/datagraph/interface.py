import typing as t
from abc import ABC, abstractmethod

import attr
from frozendict import frozendict

from ebony.time.time_range import TimeRange
from ebony.utilities.pandas_utils import IndexTensor, equals


@attr.s(frozen=True, slots=True, cache_hash=True)
class DataSetMetadata:
    """
    A `DataSetMetadata` object contains all the information to describe a dataset which
    may or may not exist.

    \TODO - attr repr method and str method are terrible for deeply nested graphs, overwrite
    """

    name: str = attr.ib()
    engine: "IPersistenceEngine" = attr.ib()

    # TODO: does static definitely belong here?
    static: bool = attr.ib()
    time_level = attr.ib()

    # TODO: restrict this t.Any to a suitable parameter-type interface, e.g. Hashable
    params: t.Dict[str, t.Any] = attr.ib(converter=frozendict)
    predecessors: t.Dict[str, "DataSetMetadata"] = attr.ib(converter=frozendict)

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


# TODO: split this into StaticDataSet and TimeSeriesDataSet?
# TODO: does this need to be hashable? It will currently fail because `data` will
#  usually be an unhashable pandas object
@attr.s(frozen=True, eq=True, hash=False)
class DataSet:
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
        return TimeRange.from_pandas(
            self.data_time_range, level=self.metadata.time_level
        )

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
