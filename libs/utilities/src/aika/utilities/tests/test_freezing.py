import pytest
from frozendict import frozendict

from aika.utilities.freezing import freeze_recursively
from aika.utilities.testing import assert_call


@pytest.mark.parametrize(
    "input, expected",
    [
        ({}, frozendict()),
        ({"A": "B"}, frozendict(A="B")),
        ({"A": {"B": 1}}, frozendict(A=frozendict(B=1))),
        ({"A": list(range(3))}, frozendict(A=tuple(range(3)))),
        ({"A": tuple(range(3))}, frozendict(A=tuple(range(3)))),
    ],
)
def test_freeze_recursively(input, expected):
    assert_call(freeze_recursively, expected, input)
