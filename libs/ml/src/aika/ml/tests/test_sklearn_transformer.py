"""
These tests check that the general structure holds for common sklearn estimators
"""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from aika.ml.interface import BivariateDataSet, SklearnEstimator


class TestSklearnEstimators:

    estimators_to_test = [
        LinearRegression(fit_intercept=True, copy_X=True),
    ]

    @pytest.mark.parametrize(
        "dataset",
        [
            BivariateDataSet(
                X=pd.DataFrame(
                    {
                        "foo": np.random.randn(100)
                        + np.array([0.01 * x for x in range(100)]),
                        "bar": np.random.randn(100),
                    }
                ),
                y=pd.Series(np.array([0.01 * x for x in range(100)])),
            )
        ],
    )
    @pytest.mark.parametrize("estimator", estimators_to_test)
    def test_basic_functions(self, dataset, estimator):
        transformer = SklearnEstimator(estimator)
        result = transformer.fit_transform(dataset)

        pickled_transformer = pickle.loads(pickle.dumps(transformer))
        pickled_result = pickled_transformer.transform(dataset)

        assert result.X.equals(pickled_result.X)
        assert result.y.equals(pickled_result.y)
