import typing as t
from abc import ABC, abstractmethod

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

    # typically defined as a constant class attribute
    time_series: bool = abstract_attribute()

    # typically specified as constructor arguments
    name: str = abstract_attribute()
    namespace: str = abstract_attribute()
    version: str = abstract_attribute()
    persistence_engine: IPersistenceEngine = abstract_attribute()

    # typically defined as properties / descriptors
    dependencies: t.Dict[str, Dependency] = abstract_attribute()
    output: DataSetMetadata = abstract_attribute()
    io_params: t.Dict[str, t.Any] = abstract_attribute()

    @abstractmethod
    def run(self):
        pass

    # TODO: have `complete` return a `CompleteResult`
    @abstractmethod
    def complete(self) -> bool:
        pass

    @abstractmethod
    def read(self, time_range: t.Optional[TimeRange] = None) -> t.Any:
        pass


class IStaticTask(ITask, ABC):

    time_series = False


class ITimeSeriesTask(ITask, ABC):

    time_series = True

    time_range: TimeRange = abstract_attribute()
    time_level: t.Union[int, str, None] = abstract_attribute()
    default_lookback: t.Optional[pd.offsets.BaseOffset] = abstract_attribute()
    completion_checker: ICompletionChecker = abstract_attribute()
