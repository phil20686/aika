import copy
import itertools
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Dict, Any, Union

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


class DataSetDeclaration:
    """
    A dataset declaration is the meta data that defines a dataset for which data may or may not actually
    exists. It is fully parameterised. It represents a dag and you can only fully declare one once you
    have created the predecessor nodes.
    """

    def __init__(
        self,
        name: str,
        time_range: TimeRange,
        params: Dict[str, any],
        predecessors: Dict[str, "DataSetDeclaration"],
    ):
        self._name = name
        self._time_range = time_range
        self._params = {
            p.name: p for p in sorted([ParamPair(n, v) for n, v in params.items()])
        }
        self._predecessors = predecessors
        self._hash = None

    def __eq__(self, other):
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
        # recurse the tree, this means hashing becomes extremely
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
    def time_range(self) -> TimeRange:
        """
        This returns the time range, if the dataset does not have a time axis this will return
        time_range(None, None).
        """
        return self._time_range

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
        return copy.copy(self._params)

    @property
    def predecessors(self) -> Dict[str, "DataSetDeclaration"]:
        """
        Returns a dict of the parameter name pointing to a different dataset declaration.
        """
        return copy.copy(self._predecessors)


class DataSet(DataSetDeclaration):
    def __init__(
        self,
        name: str,
        data: IndexTensor,
        params: Dict[str, any],
        predecessors: Dict[str, "DataSetDeclaration"],
    ):
        super().__init__(
            name=name,
            time_range=TimeRange.from_pandas(data),
            params=params,
            predecessors=predecessors,
        )
        self._data = data

    @property
    def data(self) -> IndexTensor:
        return self._data

    def update_data(self, new_data):
        """
        Creates a new dataset that has the same parameters
        but different data
        """
        return DataSet(
            name=self.name,
            time_range=TimeRange.from_pandas(new_data),
            data=new_data,
            params={x.name: x.value for x in self._params},
            predecessors=self.predecessors,
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
