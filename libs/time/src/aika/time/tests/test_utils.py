import pandas as pd
import pytest

from aika.time.tests.utils import assert_call, assert_equal


@pytest.mark.parametrize(
    "value, expect, test_kwargs, expect_assertion_failure",
    [
        (5, 6, {}, True),
        (5, 5, {}, False),
        (
            pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="foo")),
            pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="foo")),
            {},
            False,
        ),
        (
            pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="foo")),
            pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="bar")),
            {},
            True,
        ),
        (
            pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="foo")),
            pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="bar")),
            {"check_names": False},
            False,
        ),
        (
            pd.DataFrame(1.0, columns=range(3), index=range(3)),
            pd.DataFrame(1.01, columns=range(3), index=range(3)),
            {},
            True,
        ),
        (
            pd.DataFrame(1.0, columns=range(3), index=range(3)),
            pd.DataFrame(1.01, columns=range(3), index=range(3)),
            {"atol": 0.1},
            False,
        ),
        (
            pd.DataFrame(1.0, columns=range(3), index=range(3)),
            pd.DataFrame(1.0, columns=range(4), index=range(3)),
            {},
            True,
        ),
        (
            pd.DataFrame(1.0, columns=range(3), index=range(3)),
            pd.DataFrame(1.0, columns=range(3), index=range(4)),
            {},
            True,
        ),
    ],
)
def test_assert_equals(value, expect, test_kwargs, expect_assertion_failure):
    if expect_assertion_failure:
        with pytest.raises(AssertionError):
            assert_equal(value, expect, **test_kwargs)
    else:
        assert_equal(value, expect, **test_kwargs)


@pytest.mark.parametrize(
    "func, expect, args, kwargs, test_kwargs, expect_assertion_failure",
    [
        (lambda: 5, 6, (), {}, {}, True),
        (lambda: 5, 5, (), {}, {}, False),
        (
            lambda: pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="foo")),
            pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="foo")),
            (),
            {},
            {},
            False,
        ),
        (
            lambda: pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="foo")),
            pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="bar")),
            (),
            {},
            {},
            True,
        ),
        (
            lambda: pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="foo")),
            pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3, name="bar")),
            (),
            {},
            {"check_names": False},
            False,
        ),
        (
            lambda: pd.DataFrame(1.0, columns=range(3), index=range(3)),
            pd.DataFrame(1.01, columns=range(3), index=range(3)),
            (),
            {},
            {},
            True,
        ),
        (
            lambda: pd.DataFrame(1.0, columns=range(3), index=range(3)),
            pd.DataFrame(1.01, columns=range(3), index=range(3)),
            (),
            {},
            {"atol": 0.1},
            False,
        ),
        (
            lambda: pd.DataFrame(1.0, columns=range(3), index=range(3)),
            pd.DataFrame(1.0, columns=range(4), index=range(3)),
            (),
            {},
            {},
            True,
        ),
        (
            lambda: pd.DataFrame(1.0, columns=range(3), index=range(3)),
            pd.DataFrame(1.0, columns=range(3), index=range(4)),
            (),
            {},
            {},
            True,
        ),
        (lambda a, b: a - b, 3.5, (4.0, 0.5), {}, {}, False),
        (lambda a, b: a - b, 3.5, (4.0, 1.0), {}, {}, True),
        (lambda a, b: a - b, 3.5, (), {"a": 4, "b": 0.5}, {}, False),
        (lambda a, b: a - b, 3.5, (), {"a": 4, "b": 1.0}, {}, True),
        (lambda: None, None, (), {}, {}, False),
    ],
)
def test_assert_call(func, expect, args, kwargs, test_kwargs, expect_assertion_failure):
    if expect_assertion_failure:
        with pytest.raises(AssertionError):
            assert_call(func, expect, test_kwargs=test_kwargs, *args, **kwargs)
    else:
        assert_call(func, expect, test_kwargs=test_kwargs, *args, **kwargs)
