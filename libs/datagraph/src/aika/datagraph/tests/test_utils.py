import pytest as pytest
from frozendict import frozendict

from aika.utilities.testing import assert_call

from aika.datagraph.utils import normalize_parameters


@pytest.mark.parametrize(
    "input, expect",
    [
        (
            {"foo": 1.0},
            frozendict({"foo": 1.0}),
        ),
        (
            {"foo": 1},
            frozendict({"foo": 1}),
        ),
        (
            frozendict({"foo": 1}),
            frozendict({"foo": 1}),
        ),
        ({"foo": list("BAC")}, frozendict({"foo": tuple(list("BAC"))})),
        ({"foo": tuple("BAC")}, frozendict({"foo": tuple("BAC")})),
        (
            {"foo": [(1, 2), [1.0, 2.0]], "bar": ((1, 2), (1.0, 2.0))},
            frozendict({"foo": ((1, 2), (1.0, 2.0)), "bar": ((1, 2), (1.0, 2.0))}),
        ),
        (
            {"foo": {"bar": [1, 2, [3, 4]]}},
            frozendict({"foo": frozendict({"bar": (1, 2, (3, 4))})}),
        ),
        ({1, 2, 3}, ValueError("Dataset metadata params included a param")),
    ],
)
def test_normalize_parameters(input, expect):
    assert_call(normalize_parameters, expect, input)
