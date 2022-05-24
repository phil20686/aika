import inspect
import typing as t
from abc import ABC, abstractmethod
from functools import cached_property, partial
from pprint import pformat

import attr
import pandas as pd
from frozendict import frozendict

from aika.datagraph.interface import DataSetMetadata, IPersistenceEngine
from aika.time.time_range import TimeRange
from aika.utilities.pandas_utils import IndexTensor

from aika.dux.completion_checking import _infer_inherited_completion_checker
from aika.dux.interface import (
    Dependency,
    ICompletionChecker,
    IStaticTask,
    ITask,
    ITimeSeriesTask,
)


@attr.s(frozen=True)
class TimeSeriesTask(ITimeSeriesTask, ABC):

    time_range: TimeRange = attr.ib()
    name: str = attr.ib()
    namespace: str = attr.ib()
    version: str = attr.ib()
    persistence_engine: IPersistenceEngine = attr.ib()
    time_level = attr.ib(default=None)
    default_lookback: t.Optional[pd.offsets.BaseOffset] = attr.ib(default=None)
    _completion_checker: t.Optional[ICompletionChecker] = attr.ib(default=None)

    @cached_property
    def completion_checker(self):
        if self._completion_checker is not None:
            return self._completion_checker
        else:
            return _infer_inherited_completion_checker(self.dependencies)

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


@attr.s(frozen=True)
class StaticTask(IStaticTask, ABC):
    name: str = attr.ib()
    namespace: str = attr.ib()
    version: str = attr.ib()
    persistence_engine: IPersistenceEngine = attr.ib()

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


@attr.s(frozen=True)
class FunctionWrapperMixin(ITask, ABC):

    function: t.Callable[..., t.Any] = attr.ib()
    static_kwargs: frozendict[str, t.Any] = attr.ib()
    data_kwargs_deps: frozendict[str, Dependency] = attr.ib()

    @cached_property
    def dependencies(self) -> "t.Dict[str, Dependency]":
        return self.data_kwargs_deps

    def run(self):
        task_data_kwargs = self.get_data_kwargs()

        kwargs = self.static_kwargs | task_data_kwargs

        result = self.function(**kwargs)

        self.write_data(result)

    @abstractmethod
    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        pass

    @abstractmethod
    def write_data(self, data: t.Any) -> None:
        pass


@attr.s(frozen=True)
class TimeSeriesFunctionWrapper(FunctionWrapperMixin, TimeSeriesTask):
    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        return {
            name: dep.read(
                downstream_time_range=self.time_range,
                default_lookback=self.default_lookback,
            )
            for name, dep in self.data_kwargs_deps.items()
        }

    def write_data(self, data: IndexTensor) -> None:
        self.output.append(data=data, declared_time_range=self.time_range)


@attr.s(frozen=True)
class StaticFunctionWrapper(FunctionWrapperMixin, StaticTask):
    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        return {name: dep.read() for name, dep in self.data_kwargs_deps.items()}

    def write_data(self, data: t.Any) -> None:
        self.output.replace(data=data, declared_time_range=None)


def task(function, *args, time_series: bool, **kwargs):
    cls = TimeSeriesFunctionWrapper if time_series else StaticFunctionWrapper
    cls_sig = inspect.signature(cls)
    func_sig = inspect.signature(function)

    # note that it is fine for cls_kwargs and func_kwargs to intersect
    cls_kwargs = {k: v for k, v in kwargs.items() if k in cls_sig.parameters}
    func_kwargs = {k: v for k, v in kwargs.items() if k in func_sig.parameters}
    remaining_kwargs = set(kwargs).difference(cls_kwargs).difference(func_kwargs)

    if remaining_kwargs:
        raise ValueError(
            f"Got unexpected keyword arguments {pformat(remaining_kwargs)}, which are "
            f"not in the signature of function {function.__qualname__}, nor of the "
            f"task class {cls.__name__}"
        )

    bound_func_kwargs = func_sig.bind(*args, **func_kwargs).arguments

    static_kwargs = {}
    data_kwargs_deps = {}
    for key, value in bound_func_kwargs.items():
        if isinstance(value, Dependency):
            data_kwargs_deps[key] = value
        elif isinstance(value, ITask):
            dep = Dependency(
                task=value,
                lookback=None,
                inherit_frequency=None,
            )
            data_kwargs_deps[key] = dep
        else:
            static_kwargs[key] = value

    # noinspection PyArgumentList
    return cls(
        function=function,
        static_kwargs=frozendict(static_kwargs),
        data_kwargs_deps=frozendict(data_kwargs_deps),
        **cls_kwargs,
    )


time_series_task = partial(task, time_series=True)
static_task = partial(task, time_series=False)
