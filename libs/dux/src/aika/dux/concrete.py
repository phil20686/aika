import inspect
import typing as t
from abc import ABC, abstractmethod
from functools import cached_property, partial
from pprint import pformat

import attr
import pandas as pd
from frozendict import frozendict

from aika.datagraph.interface import DataSetMetadata, IPersistenceEngine
from aika.dux.completion_checking import infer_inherited_completion_checker
from aika.dux.interface import (
    Dependency,
    ITask,
    ICompletionChecker,
    ITimeSeriesTask,
    IStaticTask,
)

from aika.time.time_range import TimeRange
from aika.utilities.abstract import abstract_attribute
from aika.utilities.pandas_utils import IndexTensor


class TaskBase(ITask, ABC):
    def read(self, time_range: t.Optional[TimeRange] = None) -> t.Any:
        return self.output.read(time_range=time_range)


class TimeSeriesTaskBase(ITimeSeriesTask, TaskBase, ABC):

    _completion_checker: t.Optional[ICompletionChecker] = abstract_attribute()

    @cached_property
    def completion_checker(self):
        if self._completion_checker is not None:
            return self._completion_checker
        else:
            return infer_inherited_completion_checker(self)

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

    function: t.Callable[..., t.Any] = abstract_attribute()
    scalar_func_kwargs: frozendict[str, t.Any] = abstract_attribute()
    func_dependencies: frozendict[str, Dependency] = abstract_attribute()

    @cached_property
    def dependencies(self) -> "t.Dict[str, Dependency]":
        return self.func_dependencies

    def run(self):
        data_func_kwargs = self.get_data_kwargs()

        func_kwargs = self.scalar_func_kwargs | data_func_kwargs

        result = self.function(**func_kwargs)

        self.write_data(result)

    @abstractmethod
    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        pass

    @abstractmethod
    def write_data(self, data: t.Any) -> None:
        pass


@attr.s(frozen=True, kw_only=True)
class TimeSeriesFunctionWrapper(FunctionWrapperMixin, TimeSeriesTaskBase):

    name: str = attr.ib()
    namespace: str = attr.ib()
    version: str = attr.ib()
    persistence_engine: IPersistenceEngine = attr.ib()

    time_range: TimeRange = attr.ib()
    time_level: t.Union[int, str, None] = attr.ib(default=None)
    default_lookback: t.Optional[pd.offsets.BaseOffset] = attr.ib(default=None)
    _completion_checker: ICompletionChecker = attr.ib(default=None)

    function: t.Callable[..., t.Any] = attr.ib()
    scalar_func_kwargs: frozendict[str, t.Any] = attr.ib()
    func_dependencies: frozendict[str, Dependency] = attr.ib()

    @cached_property
    def io_params(self):
        return frozendict(
            (k, v) for k, v in self.scalar_func_kwargs.items() if k != "time_range"
        )

    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        return {
            name: dep.read(
                downstream_time_range=self.time_range,
                default_lookback=self.default_lookback,
            )
            for name, dep in self.func_dependencies.items()
        }

    def write_data(self, data: IndexTensor) -> None:
        self.output.append(data=data, declared_time_range=self.time_range)


@attr.s(frozen=True, kw_only=True)
class StaticFunctionWrapper(FunctionWrapperMixin, StaticTaskBase):

    name: str = attr.ib()
    namespace: str = attr.ib()
    version: str = attr.ib()
    persistence_engine: IPersistenceEngine = attr.ib()

    function: t.Callable[..., t.Any] = attr.ib()
    scalar_func_kwargs: frozendict[str, t.Any] = attr.ib()
    func_dependencies: frozendict[str, Dependency] = attr.ib()

    @cached_property
    def io_params(self):
        return self.scalar_func_kwargs

    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        return {name: dep.read() for name, dep in self.func_dependencies.items()}

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

    scalar_func_kwargs = {}
    func_dependencies = {}

    for key, value in bound_func_kwargs.items():
        if isinstance(value, Dependency):
            func_dependencies[key] = value
        elif isinstance(value, ITask):
            dep = Dependency(
                task=value,
                lookback=None,
                inherit_frequency=None,
            )
            func_dependencies[key] = dep
        else:
            scalar_func_kwargs[key] = value

    # noinspection PyArgumentList
    return cls(
        function=function,
        scalar_func_kwargs=frozendict(scalar_func_kwargs),
        func_dependencies=frozendict(func_dependencies),
        **cls_kwargs,
    )


time_series_task = partial(task, time_series=True)
static_task = partial(task, time_series=False)
