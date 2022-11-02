from typing import Dict, Iterator, Optional, Tuple, Union

import pandas as pd

from aika.ml.interface import BivariateDataSet
from aika.time.alignment import causal_match
from aika.utilities.pandas_utils import Tensor, Tensor2
from aika.utilities.validators import Validators


class Indexer:
    """
    Base class for generating a series of integer indexes
    that will slice a pandas index.
    """

    def __init__(self, *, index: pd.Index):
        self.index = index

    def __len__(self):
        return len(self.index)

    @property
    def batches(self) -> Iterator[Tuple[int, int]]:
        raise NotImplementedError


class Indexers:
    class All(Indexer):
        """
        returns the whole dataset.
        """

        @property
        def batches(self) -> Iterator[Tuple[int, int]]:
            yield (0, len(self.index))

    class SequentialSteps(Indexer):
        """
        Steps through the data in disjoint steps of size `step_size`.
        The final chunk will be included only if it is larger than
        min_periods.
        """

        def __init__(
            self, *, index: pd.Index, window_size: int, min_periods: Union[int, None]
        ):
            super().__init__(index=index)
            self.window_size = window_size
            self.min_periods = 0 if min_periods is None else min_periods
            if len(self) < self.min_periods:
                raise ValueError(
                    "This will produce no batches since index length < min_periods"
                )

        @property
        def batches(self) -> Iterator[Tuple[int, int]]:
            points = list(range(0, len(self), self.window_size))
            if len(self) - points[-1] > self.min_periods:
                points.append(len(self))
            for start, end in zip(points[:-1], points[1:]):
                yield (start, end)

    class RollingSteps(Indexer):
        """
        Walks forward in increments of `step_size`, expanding the window to `window_size`,
        starting with a batch of `min_periods`. if `strict` is set to true, the final
        window will not exist unless (len - min_periods) % step_size == 0. I.e. it only
        takes whole steps. If strict is false, there will be a final window even if it does
        not take an entire step size.
        """

        def __init__(
            self,
            index: pd.Index,
            window_size: int,
            step_size: int,
            min_periods: Union[int, None],
            strict: bool = False,
        ):
            super().__init__(index=index)
            self.window_size = Validators.Integers.greater_than_zero(
                window_size, "window_size"
            )
            self.step_size = Validators.Integers.greater_than_zero(
                step_size, "step_size"
            )
            self.min_periods = (
                0
                if min_periods is None
                else Validators.Integers.greater_than_zero(min_periods, "min_periods")
            )
            self.strict = strict

            if len(index) <= min_periods:
                raise ValueError("min_periods must be geq than index length")
            if self.min_periods > self.window_size:
                raise ValueError("min_periods must leq than batch size")

        @property
        def batches(self) -> Iterator[Tuple[int, int]]:
            points = list(range(self.min_periods, len(self), self.step_size))
            if not self.strict or (len(self) - points[-1]) == self.step_size:
                points.append(len(self))
            for end in points:
                yield (max(end - self.window_size, 0), end)


class CausalDataSetGenerator:
    """
    This will generate a sequence of datasets that have been causally aligned
    and which walk forward through time using the logic of one of the batchers
    above. A batcher is selected using the following recipe:
    1) If the window_size is none, will use Indexers.All which returns one dataset
    which is the whole dataset.
    2) If window_size is an int and step_size is None, then it will use the logic
    of Indexers.SequentialSteps to split the data into none overlapping subsets, if
    the minimum periods is also set then it will discard the final batch of data if
    it contains less than min_periods.
    3) If window_size and step_size are ints, then it will use the logic of
    Indexers.RollingWindows, if min_periods is set then that will be the size of the first
    dataset, else it will be window_size. That allows "expanding" from smaller datasets.
    If `strict_step_size` is set to true, then the last window will be included only if
    the data set is an exact number of step sizes so that the step is complete.
    """

    def select_indexer(self) -> Indexer:
        if self._window_size is None:
            return Indexers.All(index=self._features.index)
        elif self._step_size is None:
            return Indexers.SequentialSteps(
                index=self._features.index,
                window_size=self._window_size,
                min_periods=self._window_size
                if (self._min_periods is None and self._strict_step_size)
                else self._min_periods,
            )
        else:
            return Indexers.RollingSteps(
                index=self._features.index,
                window_size=self._window_size,
                min_periods=self._min_periods,
                step_size=self._step_size,
                strict=self._strict_step_size,
            )

    def __init__(
        self,
        *,
        features: Tensor,
        responses: Tensor2,
        window_size: Optional[int],
        min_periods: Optional[int] = None,
        step_size: Optional[int] = None,
        strict_step_size: bool = True,
        causal_kwargs: Optional[Dict] = None
    ):
        if features.empty or responses.empty:
            raise ValueError("Both features and responses need to be non empty")
        if window_size is None and not (min_periods is None and step_size is None):
            raise ValueError(
                "The arguments min_periods and step_size should be None if window_size is None"
            )

        self._responses = responses
        self._features = causal_match(data=features, index=responses, **causal_kwargs)
        self._window_size = window_size
        self._min_periods = min_periods
        self._step_size = step_size
        self._strict_step_size = strict_step_size
        self._causal_kwargs = causal_kwargs
        self._index = self.select_indexer()

    def __iter__(self) -> Iterator[BivariateDataSet]:
        for start, end in self._index.batches:
            yield BivariateDataSet(
                X=self._features.iloc[start:end], y=self._responses.iloc[start:end]
            )
