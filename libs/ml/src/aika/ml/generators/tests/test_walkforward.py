import pandas as pd
import pytest as pytest

from aika.ml.generators.walkforward import CausalDataSetGenerator
from aika.ml.interface import Dataset
from aika.utilities.testing import assert_equal, assert_error_or_return


class Indexes:

    daily = pd.date_range(start="2000-01-01", periods=14, freq="D")
    bdays = pd.date_range(start="2000-01-01", periods=10, freq="B")


@pytest.mark.parametrize(
    "causal_dataset, expected",
    [
        # All Indexer
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=None,
                causal_kwargs={"contemp": True},
            ),
            [
                Dataset(
                    X=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                )
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=None,
                causal_kwargs={"contemp": False},
            ),
            [
                Dataset(
                    X=pd.Series(
                        [x - 1 for x in Indexes.bdays.day], index=Indexes.bdays
                    ),
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                )
            ],
        ),
        # Sequential steps
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=4,
                step_size=None,
                min_periods=None,
                strict_step_size=True,
                causal_kwargs={"contemp": True},
            ),
            [
                Dataset(
                    X=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 4), (4, 8)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=4,
                step_size=None,
                min_periods=None,
                strict_step_size=True,
                causal_kwargs={"contemp": False},
            ),
            [
                Dataset(
                    X=pd.Series(
                        [x - 1 for x in Indexes.bdays.day], index=Indexes.bdays
                    ).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 4), (4, 8)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=4,
                step_size=None,
                min_periods=None,
                strict_step_size=False,
                causal_kwargs={"contemp": True},
            ),
            [
                Dataset(
                    X=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 4), (4, 8), (8, 10)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=4,
                step_size=None,
                min_periods=None,
                strict_step_size=False,
                causal_kwargs={"contemp": False},
            ),
            [
                Dataset(
                    X=pd.Series(
                        [x - 1 for x in Indexes.bdays.day], index=Indexes.bdays
                    ).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 4), (4, 8), (8, 10)]
            ],
        ),
        # Rolling Steps Indexer
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=1,
                min_periods=8,
                causal_kwargs={"contemp": True},
            ),
            [
                Dataset(
                    X=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 8), (1, 9), (2, 10)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=1,
                min_periods=8,
                causal_kwargs={"contemp": False},
            ),
            [
                Dataset(
                    X=pd.Series(
                        [x - 1 for x in Indexes.bdays.day], index=Indexes.bdays
                    ).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 8), (1, 9), (2, 10)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=2,
                min_periods=8,
                causal_kwargs={"contemp": True},
            ),
            [
                Dataset(
                    X=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 8), (2, 10)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=2,
                min_periods=8,
                causal_kwargs={"contemp": False},
            ),
            [
                Dataset(
                    X=pd.Series(
                        [x - 1 for x in Indexes.bdays.day], index=Indexes.bdays
                    ).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 8), (2, 10)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=2,
                min_periods=6,
                causal_kwargs={"contemp": True},
            ),
            [
                Dataset(
                    X=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 6), (0, 8), (2, 10)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=2,
                min_periods=6,
                causal_kwargs={"contemp": False},
            ),
            [
                Dataset(
                    X=pd.Series(
                        [x - 1 for x in Indexes.bdays.day], index=Indexes.bdays
                    ).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 6), (0, 8), (2, 10)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=2,
                min_periods=7,
                strict_step_size=True,
                causal_kwargs={"contemp": True},
            ),
            [
                Dataset(
                    X=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 7), (1, 9)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=2,
                min_periods=7,
                strict_step_size=True,
                causal_kwargs={"contemp": False},
            ),
            [
                Dataset(
                    X=pd.Series(
                        [x - 1 for x in Indexes.bdays.day], index=Indexes.bdays
                    ).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 7), (1, 9)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=2,
                min_periods=7,
                strict_step_size=False,
                causal_kwargs={"contemp": True},
            ),
            [
                Dataset(
                    X=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 7), (1, 9), (2, 10)]
            ],
        ),
        (
            CausalDataSetGenerator(
                features=pd.Series(Indexes.daily.day, index=Indexes.daily),
                responses=pd.Series(Indexes.bdays.day, index=Indexes.bdays),
                window_size=8,
                step_size=2,
                min_periods=7,
                strict_step_size=False,
                causal_kwargs={"contemp": False},
            ),
            [
                Dataset(
                    X=pd.Series(
                        [x - 1 for x in Indexes.bdays.day], index=Indexes.bdays
                    ).iloc[start:end],
                    y=pd.Series(Indexes.bdays.day, index=Indexes.bdays).iloc[start:end],
                )
                for start, end in [(0, 7), (1, 9), (2, 10)]
            ],
        ),
    ],
)
def test_causal_dataset(causal_dataset, expected):
    assert_equal(list(causal_dataset), expected)


@pytest.mark.parametrize(
    "features, responses, causal_kwargs, window_size, min_periods, step_size, strict_step_size, expect",
    [
        (
            pd.Series(dtype=float),
            pd.Series(Indexes.bdays.day, index=Indexes.bdays),
            {"contemp": True},
            None,
            None,
            None,
            True,
            ValueError("Both features and responses need to be non empty"),
        ),
        (
            pd.Series(Indexes.daily.day, index=Indexes.daily),
            pd.Series(dtype=float),
            {"contemp": True},
            None,
            None,
            None,
            True,
            ValueError("Both features and responses need to be non empty"),
        ),
        (
            pd.Series(Indexes.daily.day, index=Indexes.daily),
            pd.Series(Indexes.bdays.day, index=Indexes.bdays),
            {"contemp": True},
            None,
            10,
            None,
            True,
            ValueError("The arguments min_periods and step_size should be None"),
        ),
        (
            pd.Series(Indexes.daily.day, index=Indexes.daily),
            pd.Series(Indexes.bdays.day, index=Indexes.bdays),
            {"contemp": True},
            None,
            None,
            10,
            True,
            ValueError("The arguments min_periods and step_size should be None"),
        ),
        (
            pd.Series(Indexes.daily.day, index=Indexes.daily),
            pd.Series(Indexes.bdays.day, index=Indexes.bdays),
            {"contemp": True},
            100,
            50,
            None,
            True,
            ValueError("This will produce no batches"),
        ),
        (
            pd.Series(Indexes.daily.day, index=Indexes.daily),
            pd.Series(Indexes.bdays.day, index=Indexes.bdays),
            {"contemp": True},
            100,
            50,
            2,
            True,
            ValueError("min_periods must be geq than index length"),
        ),
        (
            pd.Series(Indexes.daily.day, index=Indexes.daily),
            pd.Series(Indexes.bdays.day, index=Indexes.bdays),
            {"contemp": True},
            5,
            8,
            2,
            True,
            ValueError("min_periods must leq than batch size"),
        ),
    ],
)
def test_error_conditions(
    features,
    responses,
    window_size,
    min_periods,
    step_size,
    strict_step_size,
    causal_kwargs,
    expect,
):
    assert_error_or_return(
        CausalDataSetGenerator,
        expect,
        features=features,
        responses=responses,
        window_size=window_size,
        min_periods=min_periods,
        step_size=step_size,
        strict_step_size=strict_step_size,
        causal_kwargs=causal_kwargs,
    )
