import pytest
import pytest as pytest

from aika.time.tests.utils import assert_call

from aika.datagraph.persistence.composite_engine import CompositeEngine
from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
from aika.datagraph.tests.persistence_tests import (
    leaf1,
    leaf1_extended,
    leaf1_final,
    leaf1_with_nan,
)
from aika.datagraph.tests.test_interface import (
    _assert_engine_contains_expected,
    _replace_engine,
    engine_generators,
)


@pytest.mark.parametrize("read_only_engine_generator", engine_generators)
@pytest.mark.parametrize("writeable_engine_generator", [HashBackedPersistanceEngine])
@pytest.mark.parametrize(
    "preload_datasets, operation_name, operation_kwargs, operation_expected, writeable_engine_expect",
    [
        # (
        #     [leaf1],
        #     "exists",
        #     {"metadata": leaf1.metadata},
        #     True,
        #     set()
        # ),
        # (
        #     [],
        #     "exists",
        #     {"metadata": leaf1.metadata},
        #     False,
        #     set()
        # ),
        # (
        #     [leaf1],
        #     "idempotent_insert",
        #     {"dataset": leaf1},
        #     True,
        #     set()
        # ),
        # (
        #     [],
        #     "idempotent_insert",
        #     {"dataset": leaf1},
        #     False,
        #     {leaf1}
        # )
        # (
        #     [leaf1],
        #     "replace",
        #     {"dataset": leaf1},
        #     True, # should it be this?
        #     {leaf1}
        # ),
        # ( # no append if its identical
        #     [leaf1],
        #     "append",
        #     {"dataset": leaf1},
        #     True,
        #     set()
        # ),
        # ( #
        #     [leaf1],
        #     "append",
        #     {"dataset": leaf1_extended},
        #     True,
        #     {leaf1_final}
        # )
        # (
        #     [leaf1_with_nan],
        #     "merge",
        #     {"dataset": leaf1_with_nan},
        #     True,
        #     set()
        # ),
        (
            [leaf1_with_nan],
            "merge",
            {"dataset": leaf1_extended},
            True,
            {
                leaf1_final.update(
                    leaf1_with_nan.data.combine_first(leaf1_extended.data),
                    leaf1_final.declared_time_range,
                )
            },
        )
    ],
)
def test_composite_engine(
    read_only_engine_generator,
    writeable_engine_generator,
    preload_datasets,
    operation_name,
    operation_kwargs,
    operation_expected,
    writeable_engine_expect,
):
    """
    The datasets are pushed into the read only engine with insert, then the operation
    is carried out.
    """
    read_only_engine = read_only_engine_generator()
    preload_datasets = _replace_engine(read_only_engine, preload_datasets)
    for dataset in preload_datasets:
        read_only_engine.idempotent_insert(dataset)

    writable_engine = writeable_engine_generator()
    composite_engine = CompositeEngine(
        read_only_engine=read_only_engine, writeable_engine=writable_engine
    )

    func = getattr(composite_engine, operation_name)
    for name in ["metadata", "dataset"]:
        if name in operation_kwargs:
            operation_kwargs[name] = operation_kwargs[name].replace_engine(
                composite_engine, True
            )

    assert_call(func, operation_expected, **operation_kwargs)
    _assert_engine_contains_expected(read_only_engine, preload_datasets)
    _assert_engine_contains_expected(
        composite_engine, _replace_engine(composite_engine, writeable_engine_expect)
    )
    _assert_engine_contains_expected(
        writable_engine, _replace_engine(writable_engine, writeable_engine_expect)
    )
