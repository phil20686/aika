import copy
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from aika.datagraph.interface import DataSet
from aika.time.utilities import _get_index_level
from aika.utilities.pandas_utils import Level, Tensor, Tensor2

"""
This library may seem like an unnecessary wrapper around eg sklearn, but it has three very important features:
1) Provides a wrapper for including arbitrary transforms cross tensorflow, pytorch, prophet etc on an equal footing.
2) Supports bi-variate transforms, i.e. transforms where what you do to X depends on y, such as alignment, cleaning.
3) Explicitly demands support for pickling, which is necessary for compatibility with aika.
4) Provides support for walkforward training via the dataset generators.
"""

# TODO: I hate this name, but I do not want it to conflict with DataSet from the datagraph either
class BivariateDataSet:
    def __init__(self, X: Tensor, y: Optional[Tensor2] = None):
        self._X = X
        self._y = y

    @property
    def X(self) -> Tensor:
        return self._X

    @property
    def y(self) -> Tensor2:
        return self._y

    def __eq__(self, other):
        if not (type(other) == BivariateDataSet):
            return NotImplemented
        x_equal = self._X.equals(other._X)
        if self._y is None and other._y is None:
            y_equal = True
        elif self._y is None or other._y is None:
            y_equal = False
        else:
            y_equal = self._y.equals(other._y)
        return x_equal and y_equal


class Transformer(ABC):
    """
    Any and all transformers must support pickle in both their fitted and unfitted state.
    I.e. must be able to fit, pickle, reload and then transform.
    """

    @abstractmethod
    def fit(self, dataset: BivariateDataSet) -> None:
        """
        May require only X, or both X and y to fit depending on the transformer.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: BivariateDataSet) -> BivariateDataSet:
        """
        Note that some transformations may overwrite y. Eg, transform for a regression
        will use X to generate new y, this will over-write any y that is passed in.
        """
        raise NotImplementedError

    def fit_transform(self, dataset) -> BivariateDataSet:
        """
        Convience method that just calls fit and then transform
        """
        self.fit(dataset)
        return self.transform(dataset)


class UnivariateStatelessTransformer(Transformer):
    """
    Represents a generic function that maps Tensor -> Tensor. Note that there is
    no fitting, this is for stateless transformers only. Must be picklable. Nothing
    is done on the fit step.
    """

    def __init__(self, func, *args, on_x=True, on_y=True, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._on_x = on_x
        self._on_y = on_y

    def fit(self, dataset: BivariateDataSet) -> None:
        return None

    def _transform_x(self, data: Tensor) -> Tensor:
        if self._on_x:
            return self._func(data, *self._args, **self._kwargs)
        else:
            return data

    def _transform_y(self, data: Optional[Tensor]) -> Optional[Tensor]:
        if self._on_y and data is not None:
            return self._func(data, *self._args, **self._kwargs)
        else:
            return data

    def transform(self, dataset: BivariateDataSet) -> BivariateDataSet:
        return BivariateDataSet(
            X=self._transform_x(dataset.X), y=self._transform_y(dataset.y)
        )


class BivariateStatelessTransformer(Transformer):
    """
    Maps a function whose first argument is a BiVariateDataset. For example, for dropping rows that
    have nans in either the X or the Y while preserving alignment.
    """

    def __init__(
        self,
        func: "Callable[[BivariateDataSet, ...], BivariateDataSet]",
        *args,
        **kwargs,
    ):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def fit(self, dataset: BivariateDataSet) -> None:
        return None

    def transform(self, dataset: BivariateDataSet) -> BivariateDataSet:
        return self._func(dataset, *self._args, **self._kwargs)


class SklearnEstimator(Transformer):
    """
    Should wrap any sklearn estimator that implements BaseEstimator. Note that estimators
    transform X into y so any y given to an estimator will be used for fitting but overridden
    when transforming.
    """

    def __init__(self, estimator):
        self._is_fitted = False
        self._estimator = estimator
        self._fitted_columns = None

    def fit(self, dataset) -> None:
        self._estimator.fit(X=dataset.X, y=dataset.y)
        if isinstance(dataset.y, pd.DataFrame):
            self._fitted_columns = dataset.y.columns
        self._is_fitted = True

    def transform(self, dataset: BivariateDataSet) -> BivariateDataSet:
        if self._is_fitted:
            array = self._estimator.predict(X=dataset.X)
            if len(array.shape) == 1 or (len(array.shape) == 2 and array.shape[1] == 1):
                # its a series
                return BivariateDataSet(
                    X=dataset.X, y=pd.Series(data=array, index=dataset.X.index)
                )
            elif len(array.shape) == 2:
                return BivariateDataSet(
                    X=dataset.X,
                    y=pd.DataFrame(
                        array, index=dataset.X.index, columns=self._fitted_columns
                    ),
                )
            else:
                raise ValueError(
                    f"Estimators must return tensors of rank one or 2 but received shape {array.shape}"
                )  # pragma: no cover
        else:
            raise ValueError("Estimator has not been fitted")


class Pipeline:
    """
    A pipeline is just a set of steps.
    """

    def __init__(self, steps: List[Transformer]):
        self._steps = steps

    @property
    def steps(self):
        return self._steps

    def fit_transform(self, dataset: BivariateDataSet):
        for step in self.steps:
            dataset = step.fit_transform(dataset)
        return dataset

    def transform(self, dataset: BivariateDataSet):
        for step in self.steps:
            dataset = step.transform(dataset)
        return dataset

    def fit(self, dataset: BivariateDataSet):
        self.fit_transform(dataset)

    def apply_to_dataset_generator(
        self, dataset_generator, time_level: Optional[Level] = None
    ) -> pd.Series:
        """
        This will repeatedly copy the pipeline and apply each copy to one of the datasets generated, it will then
        return a series of trained pipelines timestamped by the last datapoint in the training set.

        Parameters
        ----------
        dataset_generator: CausalDataSetGenerator
            The dataset generator.

        time_level: Optional[Level]
            If the dataset generator returns multi-indexed data, which level of the index is to be used
            as the timestamp for the returned models.
        Returns
        -------
        pd.Series : A series where the index is the timestamp of the model and the value is the trained model.
        """
        results = {}
        for dataset in dataset_generator:
            new_pipeline = copy.deepcopy(self)
            new_pipeline.fit(dataset)
            results[_get_index_level(dataset.X, level=time_level)[-1]] = new_pipeline
        return pd.Series(results)


def apply_trained_models(
    models: pd.Series,
    data: Tensor,
    time_level: Optional[Level] = None,
    contemp: bool = False,
):
    """
    Given a time indexed set of transformers or pipelines this function will apply the
    newest transformer/model to each row of the data at each point in time. Use in concert
    with Pipeline.apply_to_dataset_generator to train models at regular intervals and apply
    the correct model at each point in time.

    Parameters
    ----------
    models: pd.Series
        A series of trained pipelines or transformers.
    data: Tensor
        The dataset, only requires the X values since the models are already trained
        by assumption.
    time_level: Level
        If the data is indexed with multiple levels, which level should be used as the
        time for the perspective of choosing which model to apply.
    contemp: bool
        Whether data that is available at the same point in time as a model should
        use the new model (true) or the previous model (false). Defaults to `False`
        to avoid information leakage, but broadly speaking, assuming that the "data"
        was the features input to a causal dataset generator when training the data,
        it will be safe to use contemp=True here if contemp=False was used in training.
        I.e. if the data would have been excluded in training the new model it is safe
        to use the new model on it.

    Returns
    -------

    """
    points = list(
        np.searchsorted(
            _get_index_level(data.index, time_level),
            models.index,
            side="left" if contemp else "right",
        )
    )
    results = []
    for i, (start, end) in enumerate(zip(points, points[1:] + [data.shape[0]])):
        if start != end:
            results.append(
                models.iat[i]
                .transform(BivariateDataSet(X=data.iloc[start:end], y=None))
                .y
            )

    results = pd.concat(results, axis=0)
    return results
