import numpy as np
import pandas as pd
import pytest as pytest

from aika.utilities.pandas_utils import equals
from aika.utilities.testing import assert_call


@pytest.mark.parametrize(
    "left, right, expect",
    [
        (
            pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
        ),
        (  # fails due to type change
            pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float),
            pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            False,
        ),
        (
            pd.DataFrame([[1, np.nan, 3], [4, 5, 6], [7, 8, 9]]),
            pd.DataFrame([[1, np.nan, 3], [4, 5, 6], [7, 8, 9]]),
            True,
        ),
        (pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), True),
        (pd.Series([1, 2, 3], dtype=float), pd.Series([1, 2, 3]), False),
        (pd.Series([1, np.nan, 3]), pd.Series([1, np.nan, 3]), True),
        (7, 4, False),
        (7, 7, True),
        (np.array([1, 2, 3], dtype=float), np.array([1, 2, 3], dtype=float), True),
        (  # numpy arrays compare element wise equality, and python floats == python ints of the same value.
            np.array([1.0, 2, 3], dtype=float),
            np.array([1, 2, 3], dtype=int),
            True,
        ),
        (
            np.array([1, np.nan, 3], dtype=float),
            np.array([1, np.nan, 3], dtype=float),
            True,
        ),
        (np.array([1, 2, 3]), pd.Series([1, 2, 3]), False),
    ],
)
def test_equals(left, right, expect):
    assert_call(equals, expect, left, right)
