import numpy as np
import pandas as pd
import pytest

from aika.utilities.pandas_lin_alg import (
    full_dot_product,
    full_square_columns,
    right_dot_product,
    vector_dot_product,
    vector_habermad_product,
)
from aika.utilities.testing import assert_call


@pytest.mark.parametrize(
    "index, names, expected",
    [
        (
            list("CDE"),
            ("foo", "bar"),
            pd.MultiIndex.from_tuples(
                [
                    ("C", "C"),
                    ("C", "D"),
                    ("C", "E"),
                    ("D", "C"),
                    ("D", "D"),
                    ("D", "E"),
                    ("E", "C"),
                    ("E", "D"),
                    ("E", "E"),
                ],
                names=("foo", "bar"),
            ),
        )
    ],
)
def test_full_square_columns(index, names, expected):
    assert_call(full_square_columns, expected, index=index, names=names)


@pytest.mark.parametrize(
    "left, right, expect",
    [
        (  # basic
            pd.DataFrame([[1, 2, 3], [4, 5, 6]]),
            pd.DataFrame([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]),
            pd.DataFrame([[0.1, 0.4, 0.9], [3.6, 4.0, 4.2]]),
        ),
        (  # index name preserving
            pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=pd.RangeIndex(2, name="time")),
            pd.DataFrame(
                [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], index=pd.RangeIndex(2, name="time")
            ),
            pd.DataFrame(
                [[0.1, 0.4, 0.9], [3.6, 4.0, 4.2]], index=pd.RangeIndex(2, name="time")
            ),
        ),
        (  # column index name preserving
            pd.DataFrame(
                [[1, 2, 3], [4, 5, 6]], columns=pd.Index(["A", "B", "C"], name="foo")
            ),
            pd.DataFrame(
                [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
                columns=pd.Index(["A", "B", "C"], name="foo"),
            ),
            pd.DataFrame(
                [[0.1, 0.4, 0.9], [3.6, 4.0, 4.2]],
                columns=pd.Index(["A", "B", "C"], name="foo"),
            ),
        ),
        (  # column index name preserving
            pd.DataFrame(
                [[1, 2, 3], [4, 5, 6]], columns=pd.Index(["A", "B", "C"], name="foo")
            ),
            pd.DataFrame(
                [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
                columns=pd.Index(["A", "B", "D"], name="foo"),
            ),
            ValueError,
        ),
        (  # column index name preserving
            pd.DataFrame(
                [[1, 2, 3], [4, 5, 6]], columns=pd.Index(["A", "B", "C"], name="foo")
            ),
            pd.DataFrame(
                [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
                columns=pd.Index(["A", "B", "C"], name="bar"),
            ),
            ValueError,
        ),
    ],
)
def test_vector_habermad_product(left, right, expect):
    assert_call(
        vector_habermad_product,
        expect,
        left=left,
        right=right,
    )


@pytest.mark.parametrize(
    "left, right, expect",
    [
        (  # basic
            pd.DataFrame([[1, 2, 3], [4, 5, 6]]),
            pd.DataFrame([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]),
            pd.Series([1.4, 11.8]),
        ),
        (  # index name preserving
            pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=pd.RangeIndex(2, name="time")),
            pd.DataFrame(
                [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], index=pd.RangeIndex(2, name="time")
            ),
            pd.Series([1.4, 11.8], index=pd.RangeIndex(2, name="time")),
        ),
        (  # column index name preserving
            pd.DataFrame(
                [[1, 2, 3], [4, 5, 6]], columns=pd.Index(["A", "B", "C"], name="foo")
            ),
            pd.DataFrame(
                [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
                columns=pd.Index(["A", "B", "C"], name="foo"),
            ),
            pd.Series([1.4, 11.8]),
        ),
        (  # column index name preserving
            pd.DataFrame(
                [[1, 2, 3], [4, 5, 6]], columns=pd.Index(["A", "B", "C"], name="foo")
            ),
            pd.DataFrame(
                [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
                columns=pd.Index(["A", "B", "D"], name="foo"),
            ),
            ValueError,
        ),
        (  # column index name preserving
            pd.DataFrame(
                [[1, 2, 3], [4, 5, 6]], columns=pd.Index(["A", "B", "C"], name="foo")
            ),
            pd.DataFrame(
                [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
                columns=pd.Index(["A", "B", "C"], name="bar"),
            ),
            ValueError,
        ),
    ],
)
def test_vector_dot_product(left, right, expect):
    assert_call(
        vector_dot_product,
        expect,
        left=left,
        right=right,
    )


@pytest.mark.parametrize(
    "matrix, vector, expected",
    [
        (
            pd.DataFrame([[1.0, 0.0, 0.0, 2.0]], columns=full_square_columns([1, 2])),
            pd.DataFrame([[10.0, 20.0]], columns=[1, 2]),
            pd.DataFrame([[10.0, 40.0]], columns=[1, 2]),
        ),
        (
            pd.DataFrame(
                [
                    [np.nan, 0.0, 0.0, 2.0],
                    [1.0, np.nan, 0.0, 2.0],
                    [1.0, 0.0, np.nan, 2.0],
                    [1.0, 0.0, 0.0, np.nan],
                ],
                columns=full_square_columns([1, 2]),
            ),
            pd.DataFrame([[10.0, 20.0]] * 4, columns=[1, 2]),
            pd.DataFrame(
                [[np.nan, 40], [np.nan, 40], [10, np.nan], [10, np.nan]], columns=[1, 2]
            ),
        ),
    ],
)
def test_right_dot_multiply(matrix, vector, expected):
    assert_call(right_dot_product, expected, matrix, vector)


@pytest.mark.parametrize(
    "left, matrix, right, expect",
    [
        (
            pd.DataFrame([[1, 2, 3.0]], columns=pd.Index(["A", "B", "C"], name="foo")),
            pd.DataFrame(
                [[9, 8, 7, 6, 5, 4]],
                columns=pd.MultiIndex.from_product(
                    [list("ABC"), list("XY")], names=("foo", "bar")
                ),
            ),
            pd.DataFrame([[-1, 3]], columns=pd.Index(["X", "Y"], name="bar")),
            pd.Series([58.0]),
        )
    ],
)
def test_full_dot_product(left, matrix, right, expect):
    assert_call(full_dot_product, expect, left, matrix, right)
