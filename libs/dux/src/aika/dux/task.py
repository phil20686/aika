import inspect
import typing as t
from abc import ABC, abstractmethod
from functools import cached_property
from pprint import pformat

import attr
import pandas as pd
from frozendict import frozendict

from aika.datagraph.interface import DataSetMetadata, IPersistenceEngine
from aika.time.time_range import TimeRange
from aika.utilities.abstract import abstract_attribute
from aika.utilities.pandas_utils import IndexTensor

from aika.dux.interface import (
    Dependency,
    ICompletionChecker,
    IStaticTask,
    ITask,
    ITimeSeriesTask,
)


class TaskBase(ITask, ABC):
    def read(self, time_range: t.Optional[TimeRange] = None) -> t.Any:
        return self.output.read(time_range=time_range)


class TimeSeriesTaskBase(ITimeSeriesTask, TaskBase, ABC):
    def complete(self):
        return self.completion_checker.is_complete(
            metadata=self.output,
            target_time_range=self.time_range,
        )

    @cached_property
    def output(self):
        return DataSetMetadata(
            name=f"{self.namespace}.{self.name}",
            engine=self.persistence_engine,
            static=False,
            params=self.io_params,
            predecessors={
                name: dep.task.output for name, dep in self.dependencies.items()
            },
            time_level=self.time_level,
        )


class StaticTaskBase(IStaticTask, TaskBase, ABC):
    def complete(self):
        return self.output.exists()

    @cached_property
    def output(self):
        return DataSetMetadata(
            name=f"{self.namespace}.{self.name}",
            engine=self.persistence_engine,
            static=True,
            params=self.io_params,
            predecessors={
                name: dep.task.output for name, dep in self.dependencies.items()
            },
            time_level=None,
        )


class FunctionWrapperMixin(ITask, ABC):

    # it is assumed that these will be constructor arguments in concrete subclasses
    function: t.Callable[..., t.Any] = abstract_attribute()
    scalar_kwargs: frozendict[str, t.Any] = abstract_attribute()
    dependencies: frozendict[str, Dependency] = abstract_attribute()

    def run(self):
        data_kwargs = self.get_data_kwargs()
        func_kwargs = self.scalar_kwargs | data_kwargs
        result = self.function(**func_kwargs)
        self.write_data(result)

    @abstractmethod
    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        pass

    @abstractmethod
    def write_data(self, data: t.Any) -> None:
        pass

    def validate(self):
        self._validate_function()

    def _validate_function(self):
        inspect.signature(self.function).bind(
            **self.scalar_kwargs,
            **self.dependencies,
        )


@attr.s(frozen=True, kw_only=True)
class TimeSeriesFunctionWrapper(FunctionWrapperMixin, TimeSeriesTaskBase):

    name: str = attr.ib()
    namespace: str = attr.ib()
    version: str = attr.ib()
    persistence_engine: IPersistenceEngine = attr.ib()

    time_range: TimeRange = attr.ib()
    time_level: t.Union[int, str, None] = attr.ib(default=None)
    default_lookback: t.Optional[pd.offsets.BaseOffset] = attr.ib(default=None)
    completion_checker: ICompletionChecker = attr.ib()

    function: t.Callable[..., t.Any] = attr.ib()
    scalar_kwargs: frozendict[str, t.Any] = attr.ib()
    dependencies: frozendict[str, Dependency] = attr.ib()

    def __attrs_post_init__(self):
        self.validate()

    @cached_property
    def io_params(self):
        return frozendict(
            (k, v) for k, v in self.scalar_kwargs.items() if k != "time_range"
        )

    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        return {
            name: dep.read(
                downstream_time_range=self.time_range,
                default_lookback=self.default_lookback,
            )
            for name, dep in self.dependencies.items()
        }

    def write_data(self, data: IndexTensor) -> None:
        self.output.append(data=data, declared_time_range=self.time_range)


@attr.s(frozen=True, kw_only=True, auto_attribs=True)
class StaticFunctionWrapper(FunctionWrapperMixin, StaticTaskBase):

    name: str = attr.ib()
    namespace: str = attr.ib()
    version: str = attr.ib()
    persistence_engine: IPersistenceEngine = attr.ib()

    function: t.Callable[..., t.Any] = attr.ib()
    scalar_kwargs: frozendict[str, t.Any] = attr.ib()
    dependencies: frozendict[str, Dependency] = attr.ib()

    def __attrs_post_init__(self):
        self.validate()

    @cached_property
    def io_params(self):
        return self.scalar_kwargs

    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        return {name: dep.read() for name, dep in self.dependencies.items()}

    def write_data(self, data: t.Any) -> None:
        self.output.replace(data=data, declared_time_range=None)
