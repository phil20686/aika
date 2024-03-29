from typing import Dict, Optional, Sequence

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_index_equal, assert_series_equal


def assert_equal(value, expect, **kwargs):
    """
    Test utility for assorted types, see pd.testing.assert_frame_equal and siblings for supported keywords.

    Parameters
    ----------
    value: the value
    expect: the expectation
    kwargs: passed through depending on the type.

    Raises
    ------
    AssertionError: if not equal
    """
    if isinstance(value, pd.DataFrame):
        assert_frame_equal(value, expect, **kwargs)
    elif isinstance(value, pd.Series):
        assert_series_equal(value, expect, **kwargs)
    elif isinstance(value, pd.Index):
        assert_index_equal(value, expect, **kwargs)
    elif isinstance(value, (str, bytes)):
        # this is needed because string and bytes are sequences, else
        # the line below will infinitely recurse on a string.
        assert value == expect
    elif isinstance(value, Sequence) and isinstance(expect, Sequence):
        assert len(value) == len(expect)
        for v, e in zip(value, expect):
            assert_equal(v, e, **kwargs)
    else:
        assert value == expect


def _is_exception_type(expect):
    return isinstance(expect, type) and issubclass(expect, Exception)


def _is_exception_instance(expect):
    return isinstance(expect, Exception)


def assert_error_or_return(func, expect, *args, **kwargs):
    """
    Calls func(*args, **kwargs) and asserts that you get the expected error. If no error is specified,
    returns the value of func(*args, **kwargs)
    """
    if _is_exception_type(expect):
        with pytest.raises(expect):
            func(*args, **kwargs)
    elif _is_exception_instance(expect):
        with pytest.raises(type(expect), match=str(expect)):
            func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


def assert_call(func, expect, *args, test_kwargs: Optional[Dict] = None, **kwargs):
    val = assert_error_or_return(func, expect, *args, **kwargs)
    if not (_is_exception_type(expect) or _is_exception_instance(expect)):
        assert_equal(val, expect, **(test_kwargs or {}))
    return val
