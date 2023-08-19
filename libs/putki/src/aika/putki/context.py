import functools
import inspect
import typing as t

try:
    from functools import cached_property
except ImportError:
    # if python version < 3.8.
    from backports.cached_property import cached_property


import attr
import pandas as pd
from frozendict import frozendict

from aika.datagraph.interface import IPersistenceEngine
from aika.putki import ICompletionChecker
from aika.putki.completion_checking import CalendarChecker, IrregularChecker
from aika.putki.interface import Dependency, ITask, ITimeSeriesTask
from aika.putki.task import (
    StaticFunctionWrapper,
    TimeSeriesFunctionWrapper,
)
from aika.time.calendars import UnionCalendar
from aika.time.time_range import TimeRange


@attr.s(frozen=True, auto_attribs=True)
class Defaults:
    MISSING = object()

    version: str = MISSING
    persistence_engine: IPersistenceEngine = MISSING
    time_range: TimeRange = MISSING


class Inference:
    @classmethod
    def completion_checker(cls, predecessors: t.Mapping[str, Dependency]):
        time_series_dependencies: t.Dict[str, Dependency[ITimeSeriesTask]] = {
            name: dep for name, dep in predecessors.items() if dep.task.time_series
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
                f"Task has no dependencies to inherit its completion_checker from; "
                f"completion_checker must be specified explicitly for this task."
            )

        elif len(completion_checkers) == 1:
            (result,) = completion_checkers.values()
            return result

        elif all(
            isinstance(cc, CalendarChecker) for cc in completion_checkers.values()
        ):
            completion_checkers: t.Dict[str, CalendarChecker]
            calendar = UnionCalendar.merge(
                [cc.calendar for cc in completion_checkers.values()]
            )
            return CalendarChecker(calendar)

        elif all(
            isinstance(cc, IrregularChecker) for cc in completion_checkers.values()
        ):
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
                f"Task has inconsistent completion checkers among its dependencies; "
                f"dependencies {regular} all have {CalendarChecker.__name__} completion "
                f"checkers, while dependencies {irregular} have "
                f"{IrregularChecker.__name__} completion checkers. These cannot be "
                f"generically combined, so an inherited completion checker cannot be "
                f"inferred.\n To fix this, either specify `completion_checker` explicitly "
                f"for this task, or update the `inherit_frequency` settings on the "
                f"dependencies to ensure that the dependencies being inherited from have a "
                "are either all regular or all irregular."
            )

    @classmethod
    def _assert_not_zero(cls, values: set, param_name):
        if not values:
            raise ValueError(
                "There are no predecessor or default values from which to infer the "
                f"value of {param_name}. Please specify this parameter explicitly."
            )

    @classmethod
    def _get_values(
        cls,
        param_name: str,
        predecessors: t.Mapping[str, Dependency],
        defaults: Defaults,
    ):
        """
        Gets all the values of a param from the dependencies including the default value if
        it is not "missing".
        """
        values = {
            getattr(predecessor.task, param_name)
            for predecessor in predecessors.values()
        }
        default_value = getattr(defaults, param_name)
        if default_value is not Defaults.MISSING:
            values.add(default_value)
        cls._assert_not_zero(values, param_name)
        return values

    @classmethod
    def _intersect_time_ranges(cls, time_ranges: t.Collection[TimeRange]):
        return TimeRange(
            start=max(tr.start for tr in time_ranges),
            end=min(tr.end for tr in time_ranges),
        )

    @classmethod
    def infer_time_range(
        cls, predecessors: t.Mapping[str, Dependency], defaults: Defaults
    ) -> TimeRange:
        """
        Time range is the intersection of all dependency time ranges.
        """
        values = cls._get_values(
            "time_range",
            {k: v for k, v in predecessors.items() if v.task.time_series},
            defaults,
        )
        return cls._intersect_time_ranges(values)

    @classmethod
    def infer_not_default(
        cls,
        predecessors: t.Mapping[str, Dependency],
        defaults: Defaults,
        param_name: str,
    ):
        """
        Allows exactly one other parameter than the default value, assuming it is not missing.
        If there is a non-default parameter in the dependencies then this will result in the
        value from the dependencies overriding it. This means that you can eg, insert a new
        value for "version" and then this new value will propagate to all its dependencies.

        This allows you to easily "branch" a graph on the value of a single parameter.
        """
        values = cls._get_values(param_name, predecessors, defaults)
        if len(values) == 1:
            (value,) = values
            return value
        elif len(values) == 2:
            default_value = getattr(defaults, param_name)
            if default_value in values:
                return [value for value in values if not value == default_value][0]
        else:
            raise ValueError(
                "This context allows exactly one non-default value for "
                f"parameter {param_name}, but received {len(values)}"
            )


@attr.s(frozen=True)
class GraphContext:
    """
    This class provides a convenience wrapper for creating graphs of FunctionWrapper
    tasks. This is done with one of the two factory methods, `time_series_task` and
    `static_task`, which respectively produce instances of TimeSeriesFunctionWrapper
    and StaticFunctionWrapper.

    The features provided by Context are as follows:

    - The task factory methods take function arguments in a **kwargs catchall, and
      determines based on the type of the values which arguments are scalars vs
      dependencies. It also allows dependencies to be specified as bare ITask objects,
      and the factory will handle wrapping it into a Dependency object appropriately.

    - The context allows `version`, `persistence_engine`, `time_range`, and/or
      `completion_checker` to be optionally omitted from most tasks, instead being
      "inferred" from the corresponding values on the task's predecessors. This allows
      for subgraph-level defaults to be set by setting one of these values on a task,
      and then allowing all the tasks in the downstream subgraph to simply inherit that
      value.

    - As a base case for the inheritance-based inference, the context can also be
      configured with a struct of global "defaults" for `version`, `persistence_engine`,
      and `time_range` (note that the ommission of `completion_checker` from the global
      defaults is intentional, as `completion_checker` is always necessarily specific to
      any 'leaf' task)

    - Separately to the above, the context also handles namespace management. It has
      its own internal `namespace` attribute, which is provided as the `namespace`
      argument to the task constructor. It also provides an `extend_namespace` method,
      which returns a modified copy of self with the namespace set to the concatenation
      of the previous namespace and the user-specified suffix.

    """

    _INFER = object()

    defaults: Defaults = attr.ib()
    namespace: str = attr.ib(default=None)

    _TaskType = t.TypeVar("_TaskType", StaticFunctionWrapper, TimeSeriesFunctionWrapper)

    def extend_namespace(self, namespace) -> "GraphContext":
        """
        Returns a new context with namespace extended as {original}.{extension}
        """
        namespace = (
            namespace if self.namespace is None else f"{self.namespace}.{namespace}"
        )
        return attr.evolve(
            self,
            namespace=namespace,
        )

    def time_series_task(
        self,
        name: str,
        function: t.Callable,
        *,
        version: t.Optional[str] = _INFER,
        persistence_engine: t.Optional[IPersistenceEngine] = _INFER,
        time_range: t.Optional[TimeRange] = _INFER,
        time_level: t.Union[int, str, None] = None,
        default_lookback: t.Optional[pd.offsets.BaseOffset] = None,
        completion_checker: t.Optional[ICompletionChecker] = _INFER,
        **kwargs,
    ) -> TimeSeriesFunctionWrapper:
        return self._task(
            task_cls=TimeSeriesFunctionWrapper,
            function=function,
            func_kwargs=kwargs,
            cls_kwargs=dict(
                name=name,
                version=version,
                persistence_engine=persistence_engine,
                time_range=time_range,
                time_level=time_level,
                default_lookback=default_lookback,
                completion_checker=completion_checker,
            ),
        )

    def static_task(
        self,
        name: str,
        function: t.Callable,
        *,
        version: t.Optional[str] = _INFER,
        persistence_engine: t.Optional[IPersistenceEngine] = _INFER,
        **kwargs,
    ) -> StaticFunctionWrapper:
        return self._task(
            task_cls=StaticFunctionWrapper,
            function=function,
            func_kwargs=kwargs,
            cls_kwargs=dict(
                name=name,
                version=version,
                persistence_engine=persistence_engine,
            ),
        )

    def _task(
        self,
        task_cls: t.Type[_TaskType],
        function: t.Callable,
        func_kwargs: t.Dict[str, t.Any],
        cls_kwargs: t.Dict[str, t.Any],
    ) -> _TaskType:
        sig = inspect.signature(function)

        (
            scalar_kwargs,
            dependencies,
        ) = self._split_scalars_and_dependencies(func_kwargs)

        # infer values of missing class parameters from `self.defaults` and
        # `dependencies`.
        for param, value in cls_kwargs.items():
            if value is self._INFER:
                inference_method = self._inference_methods[param]
                cls_kwargs[param] = inference_method(dependencies)

        # update `scalar_kwargs` to include any arguments which also happen to be class
        # parameters.

        for key in set(sig.parameters).intersection(cls_kwargs).difference(func_kwargs):
            scalar_kwargs[key] = cls_kwargs[key]

        sig.bind(**scalar_kwargs, **dependencies)
        return task_cls(
            namespace=self.namespace,
            function=function,
            scalar_kwargs=frozendict(scalar_kwargs),
            dependencies=frozendict(dependencies),
            **cls_kwargs,
        )

    @staticmethod
    def _split_scalars_and_dependencies(func_kwargs: t.Dict[str, t.Any]):
        """
        Given a combined dict of function kwargs, split it into 'scalar' values and
        dependencies.

        Dependencies are detected as either Dependency objects, or instances of ITask
        subclasses. The former are included verbatim in the output, while the latter
        are wrapped into Dependencies for the output.

        Returns a pair of dicts, (scalar_func_kwargs, func_dependencies)
        """

        scalar_func_kwargs = {}
        func_dependencies = {}

        for key, value in func_kwargs.items():
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

        return scalar_func_kwargs, func_dependencies

    @cached_property
    def _inference_methods(self):
        return frozendict(
            version=functools.partial(
                Inference.infer_not_default,
                defaults=self.defaults,
                param_name="version",
            ),
            time_range=functools.partial(
                Inference.infer_time_range, defaults=self.defaults
            ),
            persistence_engine=functools.partial(
                Inference.infer_not_default,
                defaults=self.defaults,
                param_name="persistence_engine",
            ),
            completion_checker=Inference.completion_checker,
        )
