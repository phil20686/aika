from typing import Optional

import pandas as pd

from aika.utilities.pandas_utils import IndexTensor, Level


def _get_index(index_tensor: IndexTensor) -> pd.Index:
    """
    Just a unified api equivalent to pd.DataFrame.index except that it also works when given an index object directly.
    """
    if hasattr(index_tensor, "index"):
        return index_tensor.index
    else:
        return index_tensor


def _get_index_level(index_tensor: IndexTensor, level: Optional[Level] = None):
    """
    Unified API to the level values of an index that works on all kinds of pandas objects
    """
    index = _get_index(index_tensor)
    if index.nlevels > 1 and level is None:
        raise ValueError("A level must be given if passed a multi index")
    return index.get_level_values(0 if level is None else level)
