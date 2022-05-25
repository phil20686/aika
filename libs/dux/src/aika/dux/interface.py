import typing as t
from abc import ABC, abstractmethod
from functools import cached_property

import attr
import pandas as pd
from pandas._libs.tslibs import BaseOffset

from aika.datagraph.interface import DataSetMetadata, IPersistenceEngine
from aika.time.time_range import TimeRange
from aika.utilities.abstract import abstract_attribute

TaskType = t.TypeVar("TaskType", bound="ITask")


class ICompletionChecker(ABC):
    def is_complete(
        self,
        metadata: "DataSetMetadata",
        target_time_range: TimeRange,
    ) -> bool:
        pass


@attr.s
class CompleteResult:

    value: bool = attr.ib()
    reason: t.Optional[str] = attr.ib()

    def __bool__(self):
        return self.value


@attr.s(frozen=True)
class Dependency(t.Generic[TaskType]):

    task: TaskType = attr.ib()
    lookback: t.Optional[BaseOffset] = attr.ib(default=None)
    inherit_frequency: t.Optional[bool] = attr.ib(default=None)

    @property
    def time_series(self) -> bool:
        return self.task.time_series

    def read(
        self,
        downstream_time_range: t.Optional[TimeRange] = None,
        default_lookback: t.Optional[pd.offsets.BaseOffset] = None,
    ):

        if not self.task.time_series or downstream_time_range is None:
            return self.task.read()
        else:
            lookback = self.lookback if self.lookback is not None else default_lookback
            if lookback is not None:
                time_range = TimeRange(
                    downstream_time_range.start - lookback,
                    downstream_time_range.end,
                )
            else:
                time_range = downstream_time_range

            return self.task.read(time_range=time_range)


class ITask(ABC):

    time_series: bool = abstract_attribute()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def namespace(self) -> str:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    @abstractmethod
    def persistence_engine(self) -> IPersistenceEngine:
        pass

    @property
    @abstractmethod
    def dependencies(self) -> "t.Dict[str, Dependency]":
        pass

    @property
    @abstractmethod
    def output(self) -> DataSetMetadata:
        pass

    @property
    @abstractmethod
    def io_params(self) -> t.Dict:
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def dependencies_are_complete(self):
        pass

    # TODO: have `complete` return a `CompleteResult`
    @abstractmethod
    def complete(self) -> bool:
        pass

    def read(self, time_range: t.Optional[TimeRange] = None) -> t.Any:
        return self.output.read(time_range=time_range)


class IStaticTask(ITask, ABC):

    time_series = False


class ITimeSeriesTask(ITask, ABC):

    time_series = True

    @property
    @abstractmethod
    def time_range(self) -> TimeRange:
        pass

    @property
    @abstractmethod
    def time_level(self) -> t.Union[int, str, None]:
        pass

    @property
    @abstractmethod
    def default_lookback(self) -> t.Optional[pd.offsets.BaseOffset]:
        pass

    @property
    @abstractmethod
    def completion_checker(self) -> ICompletionChecker:
        pass

    def complete(self):
        return self.completion_checker.is_complete(
            metadata=self.output, target_time_range=self.time_range
        )
