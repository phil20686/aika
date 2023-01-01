import inspect
import logging
import typing as t
from abc import ABC, abstractmethod

try:
    from functools import cached_property
except ImportError:
    # if python version < 3.8.
    from backports.cached_property import cached_property


import pandas as pd
from frozendict import frozendict

from aika.datagraph.interface import DataSetMetadata, IPersistenceEngine
from aika.putki.interface import (
    Dependency,
    ICompletionChecker,
    IStaticTask,
    ITask,
    ITimeSeriesTask,
)
from aika.time.time_range import TimeRange
from aika.utilities.freezing import freeze_recursively
from aika.utilities.pandas_utils import IndexTensor, Level


class TaskBase(ITask, ABC):
    def __init__(
        self,
        *,
        name: str,
        namespace: str,
        version: str,
        persistence_engine: IPersistenceEngine,
        dependencies: t.Mapping[str, Dependency],
    ):
        self._name = name
        self._namespace = namespace
        self._version = version
        self._persistence_engine = persistence_engine
        self._dependencies = dependencies

    @property
    def name(self) -> str:
        return self._name

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def version(self) -> str:
        return self._version

    @property
    def persistence_engine(self) -> IPersistenceEngine:
        return self._persistence_engine

    @property
    def dependencies(self) -> t.Dict[str, Dependency]:
        return self._dependencies

    def read(self, time_range: t.Optional[TimeRange] = None) -> t.Any:
        return self.output.read(time_range=time_range)

    def __eq__(self, other: t.Any):
        #  Tasks are equal if they create the same data
        if isinstance(other, ITask):
            return self.output == other.output
        else:
            return NotImplementedError

    def __hash__(self):
        return self.output.__hash__()


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
            version=self.version,
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
            version=self.version,
            params=self.io_params,
            predecessors={
                name: dep.task.output for name, dep in self.dependencies.items()
            },
            time_level=None,
        )


class FunctionWrapperMixin(ITask, ABC):
    @property
    @abstractmethod
    def function(self) -> t.Callable[..., t.Any]:
        pass

    @property
    @abstractmethod
    def scalar_kwargs(self) -> frozendict:
        pass

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


class TimeSeriesFunctionWrapper(FunctionWrapperMixin, TimeSeriesTaskBase):
    @property
    def time_range(self) -> TimeRange:
        return self._time_range

    @property
    def completion_checker(self) -> ICompletionChecker:
        return self._completion_checker

    @property
    def function(self):
        return self._function

    @property
    def scalar_kwargs(self) -> frozendict:
        return self._scalar_kwargs

    @property
    def time_level(self) -> t.Optional[Level]:
        return self._time_level

    @property
    def default_lookback(self) -> t.Optional[pd.offsets.BaseOffset]:
        return self._default_lookback

    def __init__(
        self,
        *,
        name: str,
        namespace: str,
        version: str,
        persistence_engine: IPersistenceEngine,
        time_range: TimeRange,
        completion_checker: ICompletionChecker,
        function: t.Callable[..., t.Any],
        scalar_kwargs: t.Mapping[str, t.Any],
        dependencies: t.Mapping[str, Dependency],
        time_level: t.Optional[Level] = None,
        default_lookback: t.Optional[pd.offsets.BaseOffset] = None,
    ):
        super().__init__(
            name=name,
            namespace=namespace,
            version=version,
            persistence_engine=persistence_engine,
            dependencies=dependencies,
        )
        self._time_range = time_range
        self._completion_checker = completion_checker
        self._function = function
        self._scalar_kwargs = freeze_recursively(scalar_kwargs)
        self._time_level = time_level
        self._default_lookback = default_lookback
        self.validate()  # from the mixin

    def run(self):
        data_kwargs = self.get_data_kwargs()
        func_kwargs = self.scalar_kwargs | data_kwargs
        result = self.function(**func_kwargs)
        # a time series task should never write data outside of the targeted
        # time range.
        result = self.time_range.view(result, level=self.time_level)
        self.write_data(result)
        if not self.complete():
            raise ValueError(
                "The task appeared to run successfully and wrote its output, "
                "but according to its completion checker it is not complete."
            )
        else:
            logging.getLogger(__name__).info(
                f"Task {self._name} completed successfully"
            )

    @cached_property
    def io_params(self):
        return frozendict(
            (k, v) for k, v in self.scalar_kwargs.items() if k != "time_range"
        )

    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        results = {}
        for name, dep in self.dependencies.items():
            value = dep.read(
                downstream_time_range=self.time_range,
                default_lookback=self.default_lookback,
            )
            if value is None:
                raise ValueError(f"Failed to read dependency {name} - output was None")
            else:
                results[name] = value
        return results

    def write_data(self, data: IndexTensor) -> None:
        self.output.append(data=data, declared_time_range=self.time_range)


class StaticFunctionWrapper(FunctionWrapperMixin, StaticTaskBase):
    @property
    def function(self) -> t.Callable[..., t.Any]:
        return self._function

    @property
    def scalar_kwargs(self) -> frozendict:
        return self._scalar_kwargs

    def __init__(
        self,
        *,
        name: str,
        namespace: str,
        version: str,
        persistence_engine: IPersistenceEngine,
        function: t.Callable[..., t.Any],
        scalar_kwargs: t.Mapping[str, t.Any],
        dependencies: t.Mapping[str, Dependency],
    ):
        super().__init__(
            name=name,
            namespace=namespace,
            version=version,
            persistence_engine=persistence_engine,
            dependencies=dependencies,
        )
        self._function = function
        self._scalar_kwargs = freeze_recursively(scalar_kwargs)
        self.validate()  # from the mixin

    def run(self):
        data_kwargs = self.get_data_kwargs()
        func_kwargs = self.scalar_kwargs | data_kwargs
        result = self.function(**func_kwargs)
        self.write_data(result)

    @cached_property
    def io_params(self):
        return self.scalar_kwargs

    def get_data_kwargs(self) -> t.Dict[str, t.Any]:
        return {name: dep.read() for name, dep in self.dependencies.items()}

    def write_data(self, data: t.Any) -> None:
        self.output.replace(data=data, declared_time_range=None)
