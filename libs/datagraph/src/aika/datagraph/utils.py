import typing as t
from collections.abc import Mapping, Sequence, Set
from numbers import Number

from frozendict import frozendict


def normalize_parameters(obj: t.Any):
    """
    In order to be a well behaved parameter it is necessary that a parameter should be hashable. This means
    that we need to convert common parameter values into immutable versions.
    """
    if isinstance(obj, (Number, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return tuple(normalize_parameters(x) for x in obj)
    elif isinstance(obj, Mapping):
        return frozendict({k: normalize_parameters(v) for k, v in obj.items()})
    else:
        raise ValueError(
            f"Dataset metadata params included a param of type {type(obj)} which is not supported"
        )
