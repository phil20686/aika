"""
This module contains methods to make it easy to align pandas objects that have time series indexes.
"""
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from aika.time.utilities import _get_index, _get_index_level
from aika.utilities.pandas_utils import IndexTensor, Level, Tensor


def _reindex_by_level(tensor: Tensor, level: Optional[Level]) -> Tensor:
    """
    This helper function returns a copy of tesnor where the index of tensor has been reduced to the single
    level given by level.
    """
    index = _get_index_level(tensor, level)
    if index.duplicated().any():
        raise ValueError(
            "data must not contain any duplicated index values on the relevant level"
        )
    tensor = tensor.copy()
    tensor.index = index
    return tensor


def _shorten_data(tensor: Tensor, contemp: bool, last_ts: pd.Timestamp):
    """
    Removes any data from tensor that is after the last time stamp.
    """
    try:
        return tensor.loc[
            tensor.index <= last_ts if contemp else tensor.index < last_ts
        ]
    except TypeError as e:
        raise ValueError(
            "This error almost always results from mixing timezone naive and timezone aware datetimes."
        ) from e


def causal_match(
    data: Tensor,
    index: IndexTensor,
    contemp: bool = False,
    data_level: Optional[Level] = None,
    index_level: Optional[Level] = None,
    fill_limit: Optional[int] = None,
) -> Tensor:
    """
    This will align data onto an index. The index can be a dataframe or series but only the index is used. This will
    return the data aligned onto the index.

    Parameters
    ----------
    data: IndexTensor
        This is a pandas object that we want to align onto the index. The data must not contain any duplicated
        index values on the relevant level, i.e. must be unstacked.
    index: IndexTensor
        This is pandas object that represents the target index.
    contemp: bool
        This variable says whether an index value in the data index is an exact match is "causally available"
        in the alignment. Roughly equivalent to the semantics of np.searchsorted "side" semantics with True == "left".
    data_level :
        If data is multi-Indexed, the level on which to align.
    index_level :
        If index is multi-Indexed, the level on which to align.
    fill_limit :
        The number of times a single row of data in "data" can be re-used as the value for "index". E.g., if the
        index is daily and the data is weekly on monday, a value of 0 means only monday will have data.

    Returns
    -------
    IndexTensor : The data object reindexed to have the same index as `index`
    """

    target_index = _get_index_level(index, index_level)
    if target_index.empty:
        # result will be empty but this will preserve level names.
        return data.reindex(_get_index(index))
    unique_target_index = target_index.drop_duplicates()

    data = _reindex_by_level(data, data_level)

    # because of the semantics of search sorted we must remove any rows in the data that come after
    # the final entry in the target index.
    data = _shorten_data(data, contemp, unique_target_index[-1])

    data.index = unique_target_index[
        np.searchsorted(
            unique_target_index, data.index, side="left" if contemp else "right"
        ).astype(int)
    ]
    # if there were multiple rows of data that align onto the same target index value, we want to keep only
    # the final row ....
    data = data[~data.index.duplicated(keep="last")]
    # .... and reindex to put any duplication back so that the final index is identical to the target.
    data = data.reindex(
        target_index,
        copy=False,
        **(
            {
                "method": None,
            }
            if fill_limit == 0
            else {"method": "ffill", "limit": fill_limit}
        )
    )
    data.index = _get_index(index)
    return data


def causal_resample(
    data: Tensor,
    index: IndexTensor,
    agg_method: Union[str, Callable],
    contemp: bool = False,
    data_level: Optional[Level] = None,
    index_level: Optional[Level] = None,
) -> Tensor:
    """
    This will resample data onto an index. The index can be a dataframe or series but only the index is used.
    This will return the data aligned onto the index.

    Parameters
    ----------
    data: IndexTensor
        This is a pandas object that we want to align onto the index. The data must not contain any duplicated
        index values on the relevant level, i.e. must be unstacked.
    index: IndexTensor
        This is pandas object that represents the target index.
    agg_method:
        The aggregation method used in the resampling. This is most commonly "last", note that in pandas
        the semantics of "last" are the last non-nan value in the resampling window, which is different from
        getting the last row.
    contemp: bool
        This variable says whether an index value in the data index is an exact match is "causally available"
        in the alignment. Roughly equivalent to the semantics of np.searchsorted "side" semantics with True == "left".
    data_level :
        If data is multi-Indexed, the level on which to align.
    index_level :
        If index is multi-Indexed, the level on which to align.

    Returns
    -------
    IndexTensor : The data object reindexed to have the same index as `index`
    """
    target_index = _get_index_level(index, index_level)
    if target_index.empty:
        # returns an empty version of the tensor with indexes preserved.
        result = data.iloc[:0]
        result.index = target_index
        return result

    data = _reindex_by_level(data, data_level)
    unique_target_index = target_index.drop_duplicates()
    data = _shorten_data(data, contemp, unique_target_index[-1])

    grouper = np.searchsorted(
        unique_target_index, data.index, side="left" if contemp else "right"
    )

    result = data.groupby(grouper).aggregate(agg_method)
    result.index = unique_target_index[grouper].drop_duplicates()
    result = result.reindex(target_index)
    result.index = _get_index(index)
    return result
