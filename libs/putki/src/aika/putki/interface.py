import typing as t
from abc import ABC, abstractmethod

import attr
import pandas as pd
from pandas._libs.tslibs import BaseOffset

from aika.datagraph.interface import DataSetMetadata, IPersistenceEngine
from aika.time.time_range import TimeRange
from aika.utilities.pandas_utils import Level

TaskType = t.TypeVar("TaskType", bound="ITask")


class ICompletionChecker(ABC):
    def is_complete(
        self,
        metadata: "DataSetMetadata",
        target_time_range: TimeRange,
    ) -> bool:
        pass


# @attr.s
# class CompleteResult:
#
#     value: bool = attr.ib()
#     reason: t.Optional[str] = attr.ib()
#
#     def __bool__(self):
#         return self.value


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
    @property
    @abstractmethod
    def time_series(self) -> bool:
        pass

    # typically specified as constructor arguments
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

    # typically defined as properties / descriptors
    @property
    @abstractmethod
    def dependencies(self) -> t.Dict[str, Dependency]:
        pass

    @property
    @abstractmethod
    def output(self) -> DataSetMetadata:
        pass

    @property
    @abstractmethod
    def io_params(self) -> t.Dict[str, t.Any]:
        pass

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
    @property
    def time_series(self):
        return False


class ITimeSeriesTask(ITask, ABC):
    @property
    def time_series(self) -> bool:
        return True

    @property
    @abstractmethod
    def time_range(self) -> TimeRange:
        raise NotImplementedError

    @property
    @abstractmethod
    def time_level(self) -> t.Optional[Level]:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_lookback(self) -> t.Optional[pd.offsets.BaseOffset]:
        raise NotImplementedError

    @property
    @abstractmethod
    def completion_checker(self) -> ICompletionChecker:
        raise NotImplementedError
