from typing import Mapping

from frozendict import frozendict


def freeze_recursively(mapping: Mapping):
    return frozendict(
        (
            (k, freeze_recursively(v)) if isinstance(v, Mapping) else (k, v)
            for k, v in mapping.items()
        )
    )
