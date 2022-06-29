from typing import TypeVar, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, Series

# these types can be used to make functions that return the same pandas type as they are given.
Tensor = TypeVar("Tensor", DataFrame, Series)
IndexTensor = TypeVar("IndexTensor", DataFrame, Series, Index)
Level = Union[int, str]


def equals(left, right) -> bool:
    """
    General-purpose equality method which delegates to the appropriate pandas and numpy equality methods.

    Notes
    -----
    When left and right are numpy arrays, the type is ignored and element wise equality is sufficient, so floats
    are the same as ints of the same value. When dataframes are compared, the datatype does matter. So a series
    of ints is not equal to a series of floats of the same value, but a numpy array is. This is the design decision
    of those two respective libraries, this simply compares equality according to the expected syntax of the underlying
    objects.
    """
    if type(left) != type(right):
        return False
    elif isinstance(left, (pd.Series, pd.DataFrame, pd.Index)):
        return left.equals(right)
    elif isinstance(left, np.ndarray):
        return np.array_equal(left, right, equal_nan=True)
    else:
        return left == right
