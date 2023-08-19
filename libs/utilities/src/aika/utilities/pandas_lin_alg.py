"""
These routines are designed to operate directly on (time) series of matrices. So for example a covariance
matrix should be represented as one index level representing "time" and then a two level column index
representing a square matrix.

All matrices should be lexiographically ordered as a matter of convention. This allows the convenient use of
numpy.reshape. Functions that specifically depend on this will be explicitly mentioned in the comments.

All inputs to all functions must have identical indexes. It is the users responsibility to ensure this.
"""
from typing import Collection

import pandas as pd


def full_square_columns(index: Collection, names=(None, None)) -> pd.MultiIndex:
    """
    Creates a named and sorted index suitable for the columns of a square matrix from the given (unsorted) list
    of indexes.
    """
    sorted_index = list(sorted(index))
    return pd.MultiIndex.from_product([sorted_index, sorted_index], names=names)


def vector_habermad_product(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Element wise multiplication of stacked vectors.
    """
    if (
        left.columns.symmetric_difference(right.columns).size
        or left.columns.name != right.columns.name
    ):
        raise ValueError("Columns must be identical")
    return left.mul(right, axis=0)


def vector_dot_product(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    """
    Dot producto of two dataframes representing stacked vectors.
    """
    return vector_habermad_product(left, right).sum(axis=1, skipna=False)


def right_dot_product(matrix: pd.DataFrame, vector: pd.DataFrame) -> pd.DataFrame:
    """
    Matrix multiplication of stacked matrices onto stacked vectors.
    """
    return (
        matrix.multiply(vector, axis=0, level=1)
        .groupby(axis=1, level=0)
        .apply(lambda df: df.sum(skipna=False, axis=1))
    )
    # return matrix.multiply(vector, axis=0, level=1).sum(axis=1, level=0, skipna=False)


def full_dot_product(
    left: pd.DataFrame, matrix: pd.DataFrame, right: pd.DataFrame
) -> pd.Series:
    """
    Matrix multiplication of two vectors with a matrix. I.e. transpose(L) M R
    """
    return vector_dot_product(left, right_dot_product(matrix, right))
