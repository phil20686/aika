from typing import TypeVar
from pandas import Index, DataFrame, Series

# these types can be used to make functions that return the same pandas type as they are given.
Tensor = TypeVar("Tensor", DataFrame, Series)
IndexTensor = TypeVar("IndexTensor", DataFrame, Series, Index)
