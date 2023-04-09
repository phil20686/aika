from typing import Mapping

from frozendict import frozendict


def freeze_recursively(value):
    """
    Recursively walks ``Mapping``s and ``list``s and converts them to ``FrozenOrderedDict`` and ``tuples``, respectively.
    """
    if isinstance(value, Mapping):
        return frozendict(((k, freeze_recursively(v)) for k, v in value.items()))
    elif isinstance(value, list) or isinstance(value, tuple):
        return tuple(freeze_recursively(v) for v in value)
    return value


def unfreeze_recursively(value):
    """
    Returns frozen dicts to dicts because eg json parsers do not correctly understand frozen dicts
    """
    if isinstance(value, Mapping):
        return dict(((k, unfreeze_recursively(v)) for k, v in value.items()))
    return value
