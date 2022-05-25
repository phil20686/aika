import inspect
import typing as t
from abc import ABC
from functools import cached_property

import pandas as pd
from frozendict import frozendict

from aika.datagraph.interface import DataSetMetadata, IPersistenceEngine
from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
from aika.time.time_range import TimeRange

from aika.dux.completion_checking import _infer_inherited_completion_checker
from aika.dux.interface import (
    Dependency,
    ICompletionChecker,
    ITask,
    ITimeSeriesTask,
    TaskType,
)


class TimeSeriesTask(ITimeSeriesTask, ABC):
    def __init__(
        self,
        *,
        func: callable,
        time_range: TimeRange,
        name: str,
        persistence_engine: IPersistenceEngine,
        time_level=None,
        namespace: t.Optional[str] = "no_namespace",
        completion_checker: t.Optional[ICompletionChecker] = None,
        default_lookback: t.Optional[pd.offsets.BaseOffset] = None,
        version: t.Optional[str] = "none",
        **kwargs,  # the arguments to the actual function
    ):
        self._time_range = time_range
        self._time_level = time_level
        self._completion_checker = completion_checker
        self._name = name
        self._namespace = namespace
        self._default_lookback = default_lookback
        self._persistence_engine = persistence_engine
        self._version = version
        self._func = func
        self._dependencies = {}
        self._func_params = {}

        for name in inspect.signature(self._func).parameters:
            try:
                if name == "time_range":
                    self._func_params[name] = self._time_range
                    continue

                value = kwargs.pop(name)
                if isinstance(value, Dependency):
                    self._dependencies[name] = value
                elif isinstance(value, ITask):
                    self._dependencies[name] = Dependency(
                        task=value,
                        lookback=self._default_lookback,
                        inherit_frequency=None,
                    )
                else:
                    self._func_params[name] = value
            except KeyError:
                raise ValueError(f"The function argument {name} was not provided")

        if len(kwargs):
            raise ValueError(
                f"Extra arguments {', '.join(kwargs.keys())} were provided"
            )

    @property
    def time_range(self) -> TimeRange:
        return self._time_range

    @property
    def time_level(self) -> t.Union[int, str, None]:
        return self._time_level

    @property
    def default_lookback(self) -> t.Optional[pd.offsets.BaseOffset]:
        return self._default_lookback

    @cached_property
    def completion_checker(self):
        if self._completion_checker is not None:
            return self._completion_checker
        else:
            return _infer_inherited_completion_checker(self.dependencies)

    @property
    def io_params(self) -> t.Dict:

        return frozendict(
            version=self.version,
            **{k: v for k, v in self._func_params.items() if not k == "time_range"},
        )

    @property
    def dependencies(self) -> "t.Dict[str, Dependency]":
        return frozendict(**self._dependencies)

    @property
    def name(self):
        return self._name

    @property
    def namespace(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @cached_property
    def output(self):
        return DataSetMetadata(
            name=".".join([self._namespace, self._name]),
            static=False,
            params=self.io_params,
            time_level=self.time_level,
            predecessors={
                name: dep.task.output for name, dep in self._dependencies.items()
            },
            engine=self._persistence_engine,
        )

    @property
    def persistence_engine(self) -> IPersistenceEngine:
        return self._persistence_engine

    def dependencies_are_complete(self) -> bool:
        return all([dep.task.complete() for dep in self._dependencies.values()])

    def run(self):
        if self.complete():
            pass
        else:
            dependency_data = {
                name: dep.read() for name, dep in self._dependencies.items()
            }

            result = self._func(**dependency_data, **self._func_params)
            self.output.append(
                data=self.time_range.view(result, level=self.time_level),
                declared_time_range=self.time_range,
            )
        return self.output


class ResearchRunner:
    """
    This class will allow tasks to run immediately on addition. Its just a cheap
    way to make sure that we minimise the bumf.

    """

    def __init__(
        self,
        time_range: TimeRange,
        lookback: pd.offsets.BaseOffset,
        namespace: str = "research",
        engine: IPersistenceEngine = None,
        autorun=True,
    ):
        self._time_range = time_range
        self._lookback = lookback
        self._namespace = namespace
        self._engine = HashBackedPersistanceEngine() if engine is None else engine
        self._tasks = {}
        self._autorun = autorun

    def add_timeseries_task(self, name, func, **kwargs):
        new_task = TimeSeriesTask(
            func=func,
            name=name,
            time_range=kwargs.pop("time_range", self._time_range),
            namespace=kwargs.pop("namespace", self._namespace),
            persistence_engine=kwargs.pop("engine", self._engine),
            default_lookback=kwargs.pop("default_lookback", self._lookback),
            **kwargs,
        )
        self._tasks.setdefault(name, []).append(new_task)
        if self._autorun:
            new_task.run()

        return new_task

    def recursive_run(self, task):
        if not task.dependencies_are_complete():
            for dep in task.dependencies.values():
                dep.task.recursive_run()
        return task.run()
