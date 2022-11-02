import pandas as pd
import pytest

from aika.ml.generators.walkforward import CausalDataSetGenerator
from aika.ml.interface import (
    BivariateDataSet,
    BivariateStatelessTransformer,
    Pipeline,
    Transformer,
    UnivariateStatelessTransformer,
    apply_trained_models,
)
from aika.utilities.pandas_utils import Tensor
from aika.utilities.testing import assert_call, assert_equal


class TestBivariateDataSet:
    @pytest.mark.parametrize(
        "one, other, should_be_equal",
        [
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=pd.Series(2.0, range(5)),
                ),
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=pd.Series(2.0, range(5)),
                ),
                True,
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=pd.Series(2.0, range(5)),
                ),
                object(),
                False,
            ),
            (
                object(),
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=pd.Series(2.0, range(5)),
                ),
                False,
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=pd.Series(3.0, range(5)),
                ),
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=pd.Series(2.0, range(5)),
                ),
                False,
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=None,
                ),
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=pd.Series(2.0, range(5)),
                ),
                False,
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=pd.Series(2.0, range(5)),
                ),
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=None,
                ),
                False,
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=None,
                ),
                BivariateDataSet(
                    X=pd.Series(1.0, range(10)),
                    y=None,
                ),
                True,
            ),
        ],
    )
    def test_equality(self, one, other, should_be_equal):
        assert (one == other) == should_be_equal


class TestStatelessTransformers:
    @staticmethod
    def add_one(data: Tensor):
        return data + 1

    @staticmethod
    def add_one_bivariate(dataset: BivariateDataSet):
        return BivariateDataSet(
            X=dataset.X + 1, y=None if dataset.y is None else dataset.y + 1
        )

    @staticmethod
    def add(data: Tensor, value):
        return data + value

    @pytest.mark.parametrize(
        "input, transformer, output",
        [
            (
                BivariateDataSet(X=pd.Series(1.0, range(5)), y=None),
                UnivariateStatelessTransformer(
                    add_one,
                    on_x=True,
                    on_y=True,
                ),
                BivariateDataSet(X=pd.Series(2.0, range(5)), y=None),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(2.0, range(5))
                ),
                UnivariateStatelessTransformer(
                    add_one,
                    on_x=True,
                    on_y=True,
                ),
                BivariateDataSet(
                    X=pd.Series(2.0, range(5)), y=pd.Series(3.0, range(5))
                ),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(3.0, range(5))
                ),
                UnivariateStatelessTransformer(
                    add_one,
                    on_x=True,
                    on_y=False,
                ),
                BivariateDataSet(
                    X=pd.Series(2.0, range(5)), y=pd.Series(3.0, range(5))
                ),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(2.0, range(5))
                ),
                UnivariateStatelessTransformer(
                    add_one,
                    on_x=False,
                    on_y=True,
                ),
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(3.0, range(5))
                ),
            ),
            (
                BivariateDataSet(X=pd.Series(1.0, range(5)), y=None),
                UnivariateStatelessTransformer(
                    add,
                    1,
                    on_x=True,
                    on_y=True,
                ),
                BivariateDataSet(X=pd.Series(2.0, range(5)), y=None),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(2.0, range(5))
                ),
                UnivariateStatelessTransformer(
                    add,
                    1,
                    on_x=True,
                    on_y=True,
                ),
                BivariateDataSet(
                    X=pd.Series(2.0, range(5)), y=pd.Series(3.0, range(5))
                ),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(3.0, range(5))
                ),
                UnivariateStatelessTransformer(
                    add,
                    1,
                    on_x=True,
                    on_y=False,
                ),
                BivariateDataSet(
                    X=pd.Series(2.0, range(5)), y=pd.Series(3.0, range(5))
                ),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(2.0, range(5))
                ),
                UnivariateStatelessTransformer(
                    add,
                    1,
                    on_x=False,
                    on_y=True,
                ),
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(3.0, range(5))
                ),
            ),
            (
                BivariateDataSet(X=pd.Series(1.0, range(5)), y=None),
                UnivariateStatelessTransformer(add, on_x=True, on_y=True, value=1),
                BivariateDataSet(X=pd.Series(2.0, range(5)), y=None),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(2.0, range(5))
                ),
                UnivariateStatelessTransformer(add, on_x=True, on_y=True, value=1),
                BivariateDataSet(
                    X=pd.Series(2.0, range(5)), y=pd.Series(3.0, range(5))
                ),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(3.0, range(5))
                ),
                UnivariateStatelessTransformer(add, on_x=True, on_y=False, value=1),
                BivariateDataSet(
                    X=pd.Series(2.0, range(5)), y=pd.Series(3.0, range(5))
                ),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(2.0, range(5))
                ),
                UnivariateStatelessTransformer(add, on_x=False, on_y=True, value=1),
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(3.0, range(5))
                ),
            ),
            (
                BivariateDataSet(X=pd.Series(1.0, range(5)), y=None),
                BivariateStatelessTransformer(
                    add_one_bivariate,
                ),
                BivariateDataSet(X=pd.Series(2.0, range(5)), y=None),
            ),
            (
                BivariateDataSet(
                    X=pd.Series(1.0, range(5)), y=pd.Series(5.0, range(5))
                ),
                BivariateStatelessTransformer(
                    add_one_bivariate,
                ),
                BivariateDataSet(
                    X=pd.Series(2.0, range(5)), y=pd.Series(6.0, range(5))
                ),
            ),
        ],
    )
    def test_transform(self, input, transformer, output):
        assert_call(transformer.transform, output, input)
        assert_call(transformer.fit_transform, output, input)


class MockFittableModel(Transformer):
    """
    "Learns" the difference between mean X value and mean Y value adds that to the values when fitted.
    Expects both X and y to be a series if they exist.
    """

    def __init__(self):
        self._mean_values = 0.0

    def fit(self, dataset: BivariateDataSet) -> None:
        self._mean_values = dataset.y.mean() - dataset.X.mean()

    def transform(self, dataset: BivariateDataSet) -> BivariateDataSet:
        return BivariateDataSet(X=dataset.X, y=dataset.X.add(self._mean_values))


class TestPipeline:
    @pytest.mark.parametrize(
        "dataset_generator, pipeline, new_data, contemp, expected_model_outputs",
        [
            (
                CausalDataSetGenerator(
                    features=pd.Series(
                        2.0, index=pd.date_range(start="2000-01-01", periods=10)
                    ),
                    responses=pd.Series(
                        3.0, index=pd.date_range(start="2000-01-01", periods=10)
                    ),
                    causal_kwargs={"contemp": True},
                    window_size=8,
                    min_periods=8,
                    step_size=2,
                    strict_step_size=True,
                ),
                Pipeline(
                    [
                        MockFittableModel(),
                    ]
                ),
                pd.Series(10.0, index=pd.date_range(start="2000-01-05", periods=10)),
                False,
                pd.Series(11.0, index=pd.date_range(start="2000-01-09", periods=6)),
            ),
            (
                CausalDataSetGenerator(
                    features=pd.Series(
                        2.0, index=pd.date_range(start="2000-01-01", periods=10)
                    ),
                    responses=pd.Series(
                        3.0, index=pd.date_range(start="2000-01-01", periods=10)
                    ),
                    causal_kwargs={"contemp": True},
                    window_size=8,
                    min_periods=8,
                    step_size=2,
                    strict_step_size=True,
                ),
                Pipeline(
                    [
                        MockFittableModel(),
                    ]
                ),
                pd.Series(10.0, index=pd.date_range(start="2000-01-05", periods=10)),
                True,
                pd.Series(11.0, index=pd.date_range(start="2000-01-08", periods=7)),
            ),
            (  # here we check that it learns two different models, and applies them at the right time
                CausalDataSetGenerator(
                    features=pd.Series(
                        2.0, index=pd.date_range(start="2000-01-01", periods=10)
                    ),
                    responses=pd.Series(
                        [3.0] * 8 + [7] * 2,
                        index=pd.date_range(start="2000-01-01", periods=10),
                    ),
                    causal_kwargs={"contemp": True},
                    window_size=8,
                    min_periods=8,
                    step_size=2,
                    strict_step_size=True,
                ),
                Pipeline(
                    [
                        MockFittableModel(),
                    ]
                ),
                pd.Series(10.0, index=pd.date_range(start="2000-01-05", periods=10)),
                False,
                pd.Series(
                    [11.0] * 2 + [12.0] * 4,
                    index=pd.date_range(start="2000-01-09", periods=6),
                ),
            ),
            (  # here we check that it learns two different models, and applies them at the right time
                CausalDataSetGenerator(
                    features=pd.Series(
                        2.0, index=pd.date_range(start="2000-01-01", periods=10)
                    ),
                    responses=pd.Series(
                        [3.0] * 8 + [7] * 2,
                        index=pd.date_range(start="2000-01-01", periods=10),
                    ),
                    causal_kwargs={"contemp": True},
                    window_size=8,
                    min_periods=8,
                    step_size=2,
                    strict_step_size=True,
                ),
                Pipeline(
                    [
                        MockFittableModel(),
                    ]
                ),
                pd.Series(10.0, index=pd.date_range(start="2000-01-05", periods=10)),
                True,
                pd.Series(
                    [11.0] * 2 + [12.0] * 5,
                    index=pd.date_range(start="2000-01-08", periods=7),
                ),
            ),
            (  # here we check that it learns two different models, and applies them at the right time
                CausalDataSetGenerator(
                    features=pd.Series(
                        2.0, index=pd.date_range(start="2000-01-01", periods=10)
                    ),
                    responses=pd.Series(
                        [3.0] * 8 + [7] * 2,
                        index=pd.date_range(start="2000-01-01", periods=10),
                    ),
                    causal_kwargs={"contemp": True},
                    window_size=8,
                    min_periods=8,
                    step_size=2,
                    strict_step_size=True,
                ),
                Pipeline(
                    [
                        MockFittableModel(),
                    ]
                ),
                pd.Series(10.0, index=pd.date_range(start="2000-01-10", periods=10)),
                True,
                pd.Series(
                    [12.0] * 10,
                    index=pd.date_range(start="2000-01-10", periods=10),
                ),
            ),
        ],
    )
    def test_pipeline(
        self, dataset_generator, pipeline, new_data, contemp, expected_model_outputs
    ):
        models = pipeline.apply_to_dataset_generator(dataset_generator)
        model_outputs = apply_trained_models(models, new_data, contemp=contemp)
        assert_equal(model_outputs, expected_model_outputs)
