import pytest
from frozendict import frozendict

from aika.utilities.hashing import session_consistent_hash, int_to_bytes
from aika.utilities.testing import assert_call


@pytest.mark.parametrize("n", list(range(-10000, 10000, 10)))
def test_int_to_bytes(n):
    assert n == int.from_bytes(int_to_bytes(n), signed=True, byteorder="big")


@pytest.mark.parametrize(
    "input, expect",
    [
        ("foo", 4029768326041810678),
        (["foo", "bar"], 515142417914327403),
        ({"foo": "bar"}, 515142417914327403),
        ([{"foo": "bar"}, {"thing", "thing2", "thing3"}], TypeError),
        (
            # this is different from the others because the frozen set is reordered to thing3,thing2,thing
            [{"foo": "bar"}, frozenset(("thing", "thing2", "thing3"))],
            691820616192026942,
        ),
        (
            frozendict({"foo": {"bar": "thing"}, "thing2": "thing3"}),
            6750691437791897271,
        ),
        (
            {"foo": {"bar": "thing"}, "thing2": "thing3"},
            6750691437791897271,
        ),
        (True, 6155168585308777045),
        (False, 8459800709157675142),
        (123572316, 2087349988433736566),
        (12.235491, 4379290670735978278),
        (1.23905734e200, 2065935446728212616),
        (None, 3086615298961530842),
    ],
)
def test_session_consistent_hash(input, expect):
    assert_call(session_consistent_hash, expect, input)
