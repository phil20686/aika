import copy
import itertools
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Dict, Any, Union, Optional, List

import pandas as pd

from ebony.time.time_range import TimeRange
from ebony.utilities.pandas_utils import IndexTensor


@total_ordering
class ParamPair:
    """
    Helper class that supports equality and hash comparisons on key-value pairs including lists.
    """

    def __init__(self, name: str, value: Any):
        # \TODO make this support lists and dicts natively.
        self._name = name
        self._value = value
        # in some cases hashing will be expensive.
        self._hash = None

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    def _is_valid_operand(self, other):
        return hasattr(other, "name") and hasattr(other, "value")

    def __eq__(self, other):
        if self._is_valid_operand(other):
            return self.name == other.name and self.value == other.value
        else:
            return NotImplemented

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.name, self.value))
        return self._hash

    def __lt__(self, other):
        if self._is_valid_operand(other):
            return self.name.lower() < other.name.lower()
        else:
            return NotImplemented


class iCompleteCalculator:
    @property
    def name(self):
        return self._name

    def is_complete(self, target_time_range, metadata: "DataSetMetaData") -> bool:
        raise NotImplementedError


class CompletenessCalculators:
    class Names:
        """
        These are just the ones that can be given as a string argument, more complicated completeness checkers
        mean that
        """

        STATIC = "Static"
        REGULAR = "Regular"
        IRREGULAR = "Irregular"
        TOLERANCE = "Tolerance"

        # these are just the ones that
        _all_freq = set([STATIC, REGULAR, IRREGULAR])

        @classmethod
        def validate_frequency(cls, freq: str):
            if not freq in cls._all_freq:
                raise ValueError(
                    f"Unrecognised frequency {freq}, valid frequencies are {cls.all_freq}"
                )
            return freq

    class Regular:
        """
        If the end point of your declared dataset is the latest of all your incoming datasets.
        """

        def __init__(self, predecessors_to_include: Optional[List[str]] = None):
            self._name = CompletenessCalculators.Names.REGULAR
            self._predecessors_to_include = predecessors_to_include

        def is_complete(
            self, target_time_range: TimeRange, metadata: "DataSetMetaData"
        ) -> bool:
            predecessors_to_include = self._predecessors_to_include or list(
                metadata._predecessors.keys()
            )
            expected_last_data_point = max(
                [
                    metadata._predecessors[name].data_time_range.end
                    for name in predecessors_to_include
                ]
            )
            return expected_last_data_point <= metadata.data_time_range.end

    class Irregular(iCompleteCalculator):
        """
        An irregular dataset is complete if it has already been run on this declared time range.
        """

        def __init__(self):
            self._name = CompletenessCalculators.Names.IRREGULAR

        def is_complete(self, target_time_range, metadata: "DataSetMetaData") -> bool:
            return target_time_range == metadata.declared_time_range

    class Static(iCompleteCalculator):
        """
        A static dataset is complete if it exists, as definitionally it never changes.
        """

        def __init__(self):
            self._name = CompletenessCalculators.Names.STATIC

        def is_complete(self, target_time_range, metadata: "DataSetMetaData") -> bool:
            return True

    class TolerenaceCheck(iCompleteCalculator):
        def __init__(self, tolerance: pd.Timedelta):
            self._name = CompletenessCalculators.Names.TOLERANCE
            self._tolerance = tolerance

        def is_complete(self, target_time_range, metadata: "DataSetMetaData") -> bool:
            return (
                target_time_range.end - self._tolerance <= metadata.data_time_range.end
            )

    @classmethod
    def default_logic(
        cls, input: Optional[Union[str, List[str]]], metadata: "DataSetDeclaration"
    ):
        """ """
        if isinstance(input, str):
            cls.Names.validate_frequency(input)
            return getattr(cls, input)()
        elif isinstance(input, list):
            return CompletenessCalculators.Regular(input)
        elif input is None:
            # work out the frequency in the cases where you can
            predecessor_frequencies = [
                d.completeness_checker.name for d in metadata.predecessors.values()
            ]
            if len(predecessor_frequencies) == 0:
                raise ValueError(
                    "Default Logic cannot handle zero predecessors, please explicitly specify logic"
                )
            if all([f == cls.Names.STATIC for f in predecessor_frequencies]):
                return cls.Static()
            elif any([f == cls.Names.IRREGULAR for f in predecessor_frequencies]):
                return CompletenessCalculators.Irregular()
            elif all(
                [
                    f in [cls.Names.REGULAR, cls.Names.TOLERANCE]
                    for f in predecessor_frequencies
                ]
            ):
                return CompletenessCalculators.Regular()
            else:
                raise ValueError(
                    "Cannot infer the desired completeness checking behaviour"
                )


class DataSetDeclaration:
    """
    A dataset declaration is the meta data that defines a dataset for which data may or may not actually
    exists. It is fully parameterised. It represents a dag and you can only fully declare one once you
    have created the predecessor nodes.
    """

    def __init__(
        self,
        name: str,
        *,
        declared_time_range: TimeRange,
        params: Dict[str, any],
        predecessors: Dict[str, "DataSetDeclaration"],
        completeness_checker: Optional[
            Union[str, pd.Timedelta, iCompleteCalculator]
        ] = None,
        level: Union[int, str] = None,
    ):
        self._name = name
        self._level = level
        self._declared_time_range = declared_time_range
        self._params = {
            p.name: p for p in sorted([ParamPair(n, v) for n, v in params.items()])
        }
        self._predecessors = predecessors
        self._hash = None
        if isinstance(completeness_checker, iCompleteCalculator):
            self._completeness_checker = completeness_checker
        else:
            self._completeness_checker = CompletenessCalculators.default_logic(
                completeness_checker, self
            )

    def __eq__(self, other):
        """
        By design, frequency and declared time range are not considered part of the equality declaration. Changing
        the frequency may effect whether a dataset is declared complete, but it will not affect the content of that
        dataset for any actually existing row.
        """
        if isinstance(other, DataSetDeclaration):
            return (
                self.name == other.name
                and self._params == other._params
                and self._predecessors == other._predecessors
            )
        else:
            return NotImplemented

    def __hash__(self):
        # hashing is potentially extremely expensive if forced to
        # recurse the tree so we cache, and now (repeat) hashing becomes extremely
        # cheap.
        if self._hash is None:
            self._hash = hash(
                tuple(
                    itertools.chain(
                        (self._name,),
                        self._params.values(),
                        self._predecessors.values(),
                    )
                )
            )
        return self._hash

    @property
    def declared_time_range(self) -> TimeRange:
        """
        This returns the time range, if the dataset does not have a time axis this will return
        time_range(None, None).
        """
        return self._declared_time_range

    @property
    def level(self) -> Union[int, str, None]:
        """
        The level of the index from which to infer the time range and on which the data is time-ordered.
        """
        return self._level

    @property
    def completeness_checker(self) -> iCompleteCalculator:
        return self._completeness_checker

    @property
    def name(self) -> str:
        """
        Returns the name of this dataset,

        Returns
        -------

        """
        return self._name

    @property
    def params(self) -> Dict[str, ParamPair]:
        """
        Returns a dict keyed off the ParamPair name value. Ordered by name for convenience.
        """
        return {name: p.value for name, p in self._params.items()}

    @property
    def predecessors(self) -> Dict[str, "DataSetDeclaration"]:
        """
        Returns a dict of the parameter name pointing to a different dataset declaration.
        """
        return copy.copy(self._predecessors)


class DataSetMetaData(DataSetDeclaration):
    """
    If dataset declaration describes a dataset before it [is known to] exists, then dataset meta data
    describes a dataset that already exists. In particular, it has both a declared_time_range, and a data_time_range.
    """

    def __init__(
        self,
        name: str,
        *,
        declared_time_range: TimeRange,
        data_time_range: TimeRange,
        params: Dict[str, any],
        predecessors: Dict[str, "DataSetMetaData"],
        completeness_checker: Optional[
            Union[str, pd.Timedelta, iCompleteCalculator]
        ] = None,
        level: Union[int, str] = None,
    ):
        super().__init__(
            name=name,
            declared_time_range=declared_time_range,
            params=params,
            completeness_checker=completeness_checker,
            level=level,
            predecessors=predecessors,
        )
        self._data_time_range = data_time_range

    @property
    def data_time_range(self) -> TimeRange:
        return self._data_time_range

    @property
    def predecessors(self) -> Dict[str, "DataSetMetaData"]:
        """
        Returns a dict of the parameter name pointing to a different dataset declaration.
        """
        return copy.copy(self._predecessors)

    def update_data_time_range(self, data: IndexTensor) -> "DataSetMetaData":
        """
        This only updates teh time range on the dataset meta data using the TimeRange.from_pandas on the data
        that was passed.
        """
        return DataSetMetaData(
            name=self.name,
            declared_time_range=self.declared_time_range,
            completeness_checker=self.completeness_checker,
            level=self.level,
            data_time_range=TimeRange.from_pandas(data, level=self.level),
            params=self.params,
            predecessors=self.predecessors,
        )


class DataSet:
    @staticmethod
    def build(
        name: str,
        data: IndexTensor,
        *,
        params: Dict[str, any],
        predecessors: Dict[str, "DataSetMetaData"],
        declared_time_range: TimeRange = None,
        completeness_checker: Optional[
            Union[str, pd.Timedelta, iCompleteCalculator]
        ] = None,
        level: Union[int, str] = None,
    ) -> "DataSet":
        return DataSet(
            metadata=DataSetMetaData(
                name=name,
                declared_time_range=declared_time_range
                or TimeRange.from_pandas(data, level),
                data_time_range=TimeRange.from_pandas(data, level),
                params=params,
                predecessors=predecessors,
                completeness_checker=completeness_checker,
            ),
            data=data,
        )

    def __init__(
        self,
        *,
        metadata: DataSetMetaData,
        data: IndexTensor,
    ):
        self._metadata = metadata
        self._data = data

    def __hash__(self):
        return hash((self._metadata, self._metadata.data_time_range))

    def __eq__(self, other):
        if not isinstance(other, DataSet):
            return NotImplemented
        else:
            return (self.metadata == other.metadata) and (
                self.metadata.data_time_range == other.metadata.data_time_range
            )

    @property
    def metadata(self) -> DataSetMetaData:
        return self._metadata

    @property
    def data(self) -> IndexTensor:
        return self._data

    def update_data(self, new_data):
        """
        Creates a new dataset that has the same parameters
        but different data
        """
        return DataSet(
            metadata=DataSetMetaData.update_data_time_range(self._metadata, new_data),
            data=new_data,
        )


class iPersistenceEngine:
    def exists(
        self, declaration: DataSetDeclaration
    ) -> Union[None, DataSetDeclaration]:
        """
        If the dataset already exists, return the existing dataset declaration. By construction these two
        objects differ only by timerange.
        """
        raise NotImplementedError

    def replace(self, dataset: DataSet) -> bool:
        """
        Replaces any existing dataset. Returns true if a dataset already existed.
        """
        raise NotImplementedError

    def append(self, dataset: DataSet) -> bool:
        """
        Only adds new rows to the existing dataset, returns true
        if the dataset already exists.
        """
        raise NotImplementedError

    def merge(self, dataset: DataSet) -> bool:
        """
        The pandas semantics of combine_first on the existing dataset and the new dataset.
        """
        raise NotImplementedError
