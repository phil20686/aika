import numpy as np
import pandas as pd
from typing import TypeVar
from pandas import Index, DataFrame, Series

# these types can be used to make functions that return the same pandas type as they are given.
Tensor = TypeVar("Tensor", DataFrame, Series)
IndexTensor = TypeVar("IndexTensor", DataFrame, Series, Index)


def equals(left, right) -> bool:
    """
    General-purpose equality method which works properly with pandas tensors and numpy
    arrays
    """
    if type(left) != type(right):
        return False
    elif isinstance(left, (pd.Series, pd.DataFrame, pd.Series)):
        return left.equals(right)
    elif isinstance(left, np.ndarray):
        return np.array_equal(left, right, equal_nan=True)
    else:
        return left == right
