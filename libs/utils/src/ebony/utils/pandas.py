from typing import TypeVar
import pandas as pd

# these types can be used to make functions that return the same pandas type as they are given.
Tensor = TypeVar('Tensor', pd.DataFrame, pd.Series)
IndexTensor = TypeVar('IndexTensor', pd.DataFrame, pd.Series, pd.Index)