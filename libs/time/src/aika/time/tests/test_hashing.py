import pytest
from frozendict import frozendict

from aika.utilities.hashing import session_consistent_hash
from aika.utilities.testing import assert_call


@pytest.mark.parametrize(
    "input, expect",
    [
        ("foo", 111697143631036784911000410520264551402),
        (["foo", "bar"], 257831524625702821500684183509069276395),
        ({"foo": "bar"}, 257831524625702821500684183509069276395),
        ([{"foo": "bar"}, {"thing", "thing2", "thing3"}], TypeError),
        (
            [{"foo": "bar"}, frozenset(("thing", "thing2", "thing3"))],
            204689181545197912485084036022927190492,
        ),
        (
            frozendict({"foo": {"bar": "thing"}, "thing2": "thing3"}),
            318638353775312646373405049856364150438,
        ),
        (
            {"foo": {"bar": "thing"}, "thing2": "thing3"},
            318638353775312646373405049856364150438,
        ),
    ],
)
def test_session_consistent_hash(input, expect):
    assert_call(session_consistent_hash, expect, input)
