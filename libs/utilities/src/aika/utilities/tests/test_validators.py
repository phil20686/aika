import pytest as pytest

from aika.utilities.testing import assert_call
from aika.utilities.validators import Validators


class TestIntegerValidators:
    @pytest.mark.parametrize(
        "value, name, expect",
        [(1.0, "foo", ValueError("Parameter foo must be an int")), (1, "foo", 1)],
    )
    def test_check_type(self, value, name, expect):
        assert_call(Validators.Integers.check_type, expect, value, name)

    @pytest.mark.parametrize(
        "value, name, expect",
        [
            (1.0, "foo", ValueError("Parameter foo must be an int")),
            (1, "foo", 1),
            (-1, "bar", ValueError("Parameter bar must be greater than zero")),
        ],
    )
    def test_greater_than_zero(self, value, name, expect):
        assert_call(Validators.Integers.greater_than_zero, expect, value, name)
