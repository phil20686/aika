import pandas as pd
import pytest as pytest

from aika.time.time_range import TimeRange
from aika.utilities.testing import assert_call

from aika.datagraph.interface import DataSet, DataSetMetadata
from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine


@pytest.mark.parametrize(
    "name, static, version, params, predecessors, time_level, engine, expect",
    [
        (
            "foo",
            True,
            "test",
            {"bar": 1},
            {},
            "something",
            None,
            ValueError("Cannot specify a time level on static data"),
        )
    ],
)
def test_datasetmetadata_constructor(
    name, static, version, params, predecessors, time_level, engine, expect
):
    assert_call(
        DataSetMetadata,
        expect,
        name=name,
        static=static,
        params=params,
        version=version,
        predecessors=predecessors,
        time_level=time_level,
        engine=engine,
    )


@pytest.mark.parametrize(
    "metadata, data, declared_time_range, expect",
    [
        (
            DataSetMetadata(
                name="foo",
                static=True,
                version="0.0",
                params={"bar": 1.0},
                predecessors={},
                time_level=None,
                engine=HashBackedPersistanceEngine(),
            ),
            pd.Series(
                index=pd.date_range("2020-01-01", freq="D", periods=10), dtype=float
            ),
            TimeRange("2020-01-01", "2020-01-02"),
            ValueError("declared_time_range must be None for static data"),
        ),
        (
            DataSetMetadata(
                name="foo",
                static=False,
                params={"bar": 1.0},
                version="0.0",
                predecessors={},
                time_level=None,
                engine=HashBackedPersistanceEngine(),
            ),
            pd.Series(
                index=pd.date_range("2020-01-01", freq="D", periods=10), dtype=float
            ),
            None,
            ValueError("Must declare a time range"),
        ),
        (
            DataSetMetadata(
                name="foo",
                static=False,
                params={"bar": 1.0},
                version="0.0",
                predecessors={},
                time_level=None,
                engine=HashBackedPersistanceEngine(),
            ),
            pd.Series(
                index=pd.date_range("2020-01-01", freq="D", periods=10), dtype=float
            ),
            TimeRange("2020-01-01", "2020-01-02"),
            ValueError("Invalid declared_time_range"),
        ),
    ],
)
def test_dataset_constructor(metadata, data, declared_time_range, expect):
    assert_call(
        DataSet,
        expect,
        metadata=metadata,
        data=data,
        declared_time_range=declared_time_range,
    )
