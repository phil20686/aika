from typing import Dict, Optional

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
    else:
        assert value == expect


def assert_or_raise(func, expect, *args, **kwargs):
    """
    Calls func(*args, **kwargs) and asserts that you get the expected error. If no error is specified,
    returns the value of func(*args, **kwargs)
    """
    if isinstance(expect, type) and issubclass(expect, Exception):
        with pytest.raises(expect):
            func(*args, **kwargs)
    elif isinstance(expect, Exception):
        with pytest.raises(expect, match=str(expect)):
            func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


def assert_call(func, expect, *args, test_kwargs: Optional[Dict] = None, **kwargs):
    if isinstance(expect, type) and issubclass(expect, Exception):
        with pytest.raises(expect):
            func(*args, **kwargs)
    elif isinstance(expect, Exception):
        with pytest.raises(type(expect), match=str(expect)):
            func(*args, **kwargs)
    else:
        if test_kwargs is None:
            test_kwargs = {}
        val = func(*args, **kwargs)
        assert_equal(val, expect, **test_kwargs)
        return val
