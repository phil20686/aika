from abc import abstractmethod, ABC
from typing import List, Optional, Union
import pandas as pd
from aika.utilities.pandas_utils import Tensor, Tensor2

"""
This library may seem like an unnecessary wrapper around eg sklearn, but it has three very important features:
1) Provides a wrapper for including arbitrary transforms cross tensorflow, pytorch, prophet etc on an equal footing.
2) Supports bi-variate transforms, i.e. transforms where what you do to X depends on y, such as alignment, cleaning.
3) Explicitly demands support for pickling, which is necessary for compatibility with aika.
4) Provides support for walkforward training via the dataset generators.
"""


class Dataset:
    def __init__(self, X: Tensor, y: Optional[Tensor2] = None):
        self._X = X
        self._y = y

    @property
    def X(self) -> Tensor:
        return self._X

    @property
    def y(self) -> Tensor2:
        return self._y


class Transformer(ABC):
    """
    Any and all transformers must support pickle in both their fitted and unfitted state.
    I.e. must be able to fit, pickle, reload and then transform.
    """

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        """
        May require only X, or both X and y to fit depending on the transformer.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        """
        Note that some transformations may overwrite y. Eg, transform for a regression
        will use X to generate new y, this will over-write any y that is passed in.
        """
        raise NotImplementedError

    def fit_transform(self, dataset) -> Dataset:
        """
        Convience method that just calls fit and then transform
        """
        self.fit(dataset)
        return self.transform(dataset)


class GenericStatelessTransformer(Transformer):
    """
    Represents a generic function that maps Tensor -> Tensor. Note that there is
    no fitting, this is for stateless transformers only. Must be picklable.
    """

    def __init__(self, func, *args, on_x=True, on_y=True, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._on_x = on_x
        self._on_y = on_y

    def fit(self, dataset: Dataset) -> None:
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

    def transform(self, dataset: Dataset) -> Dataset:
        return Dataset(X=self._transform_x(dataset.X), y=self._transform_y(dataset.y))


class SklearnEstimator(Transformer):
    """
    Should wrap any sklearn estimator that implements BaseEstimator. Note that estimators
    transform X into y so any y given to an estimator will be used for fitting but overridden
    when transforming.
    """

    def __init__(self, estimator):
        self._is_fitted = False
        self._estimator = estimator

    def fit(self, dataset) -> None:
        self._estimator.fit(X=dataset.X, y=dataset.y)
        self._is_fitted = True

    def transform(self, dataset: Dataset) -> Dataset:
        if self._is_fitted:
            array = self._estimator.predict(X=dataset.X)
            if len(array.shape) == 1 or (len(array.shape) == 2 and array.shape[1] == 1):
                # its a series
                return Dataset(
                    X=dataset.X, y=pd.Series(data=array, index=dataset.X.index)
                )
            elif len(array.shape) == 2:
                return Dataset(
                    X=dataset.X,
                    y=pd.DataFrame(
                        array, index=dataset.X.index, columns=dataset.Y.columns
                    ),
                )
            else:
                raise ValueError(
                    f"Estimators must return tensors of rank one or 2 but recieved shape {array.shape}"
                )
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

    def fit_transform(self, dataset: Dataset):
        for step in self.steps:
            dataset = step.fit_transform(dataset)
        return dataset

    def transform(self, dataset: Dataset):
        for step in self.steps:
            dataset = step.transform(dataset)
        return dataset

    def fit(self, dataset: Dataset):
        self.fit_transform(dataset)
