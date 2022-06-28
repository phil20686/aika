import inspect
import typing as t
from abc import ABC, abstractmethod
from functools import cached_property, partial
from pprint import pformat

import attr
import pandas as pd
from frozendict import frozendict
from pandas.tseries.offsets import BaseOffset

from aika.datagraph.completion_checking import (
    CalendarChecker,
    ICompletionChecker,
    IrregularChecker,
)
from aika.datagraph.interface import DataSetMetadata, IPersistenceEngine
from aika.time.calendars import UnionCalendar
from aika.time.time_range import TimeRange
from aika.utilities.abstract import abstract_attribute
from aika.utilities.pandas_utils import IndexTensor


@attr.s
class CompleteResult:

    value: bool = attr.ib()
    reason: t.Optional[str] = attr.ib()

    def __bool__(self):
        return self.value


TaskType = t.TypeVar("TaskType", bound="ITask")


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
    def dependencies(self) -> "t.Dict[str, Dependency]":
        pass

    @property
    @abstractmethod
    def output(self) -> DataSetMetadata:
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


@attr.s(frozen=True)
class Task(ITask, ABC):

    name: str = attr.ib()
    namespace: str = attr.ib()
    version: str = attr.ib()
    persistence_engine: IPersistenceEngine = attr.ib()

    def io_params(self):
        return frozendict(
            version=self.version,
        )

    def read(self, time_range: t.Optional[TimeRange] = None) -> t.Any:
        return self.output.read(time_range=time_range)


@attr.s(frozen=True)
class TimeSeriesTask(Task, ABC):

    time_series = True
    time_range: TimeRange = attr.ib()
    time_level = attr.ib(default=None)

    default_lookback: t.Optional[pd.offsets.BaseOffset] = attr.ib(default=None)
    _completion_checker: t.Optional[ICompletionChecker] = attr.ib(default=None)

    @cached_property
    def completion_checker(self):
        if self._completion_checker is not None:
            return self._completion_checker
        else:
            return self._infer_inherited_completion_checker()

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
            params=self.io_params(),
            predecessors={
                name: dep.task.output for name, dep in self.dependencies.items()
            },
            time_level=self.time_level,
        )

    def _infer_inherited_completion_checker(self):

        time_series_dependencies: t.Dict[str, Dependency[TimeSeriesTask]] = {
            name: dep for name, dep in self.dependencies.items() if dep.task.time_series
        }

        explicit_inheritors = set()
        explicit_non_inheritors = set()

        for name, dep in time_series_dependencies.items():
            if dep.inherit_frequency:
                explicit_inheritors.add(name)
            elif dep.inherit_frequency is not None:
                explicit_non_inheritors.add(name)

        inheritors = (
            explicit_inheritors
            if explicit_inheritors
            else (set(time_series_dependencies) - explicit_non_inheritors)
        )

        completion_checkers: t.Dict[str, ICompletionChecker] = {
            name: time_series_dependencies[name].task.completion_checker
            for name in inheritors
        }

        if len(completion_checkers) == 0:
            raise ValueError(
                f"Task {self.name} in namespace {self.namespace} has no dependencies "
                "to inherit its completion_checker from; completion_checker must be "
                "specified explicitly for this task."
            )

        elif len(completion_checkers) == 1:
            (result,) = completion_checkers.values()
            return result

        elif all(isinstance(cc, CalendarChecker) for cc in completion_checkers):
            completion_checkers: t.Dict[str, CalendarChecker]
            calendar = UnionCalendar.merge(
                [cc.calendar for cc in completion_checkers.values()]
            )
            return CalendarChecker(calendar)

        elif all(isinstance(cc, IrregularChecker) for cc in completion_checkers):
            # note that this branch is technically redundant since IrregularChecker()
            # is a singleton and hence this case will always be covered by the len == 1
            # branch.
            return IrregularChecker()

        else:
            regular = {
                name
                for name, checker in completion_checkers.items()
                if isinstance(checker, CalendarChecker)
            }

            irregular = set(completion_checkers) - regular

            raise ValueError(
                f"Task {self.name} in namespace {self.namespace} has inconsistent "
                f"completion checkers among its dependencies; dependencies {regular} "
                f"all have {CalendarChecker.__name__} completion checkers, while "
                f"dependencies {irregular} have {IrregularChecker.__name__} "
                "completion checkers. These cannot be generically combined, so an "
                "inherited completion checker cannot be inferred.\n"
                "To fix this, either specify `completion_checker` explicitly for this "
                "task, or update the `inherit_frequency` settings on the dependencies "
                "to ensure that the dependencies being inherited from have a "
                "are either all regular or all irregular."
            )


@attr.s(frozen=True)
class StaticTask(Task, ABC):

    time_series = False

    def complete(self):
        return self.output.exists()

    @cached_property
    def output(self):
        return DataSetMetadata(
            name=f"{self.namespace}.{self.name}",
            engine=self.persistence_engine,
            static=True,
            params=self.io_params(),
            predecessors={
                name: dep.task.output for name, dep in self.dependencies.items()
            },
            time_level=None,
        )


@attr.s(frozen=True)
class FunctionWrapperMixin(Task, ABC):

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
