import inspect
import typing as t
from functools import cached_property

import attr
import pandas as pd
from frozendict import frozendict

from aika.datagraph.interface import IPersistenceEngine
from aika.dux import ICompletionChecker
from aika.dux.completion_checking import infer_inherited_completion_checker
from aika.dux.task import (
    TimeSeriesFunctionWrapper,
    StaticFunctionWrapper,
)
from aika.dux.interface import Dependency, ITask
from aika.time.time_range import TimeRange


@attr.s(frozen=True, auto_attribs=True)
class Defaults:

    MISSING = object()

    version: str = MISSING
    persistence_engine: IPersistenceEngine = MISSING
    time_range: TimeRange = MISSING


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

    def extend_namespace(self, namespace):
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
    ):

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
    ):

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
    ):
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
        scalar_kwargs |= {
            key: cls_kwargs[key]
            for key in (
                set(sig.parameters).intersection(cls_kwargs).difference(func_kwargs)
            )
        }

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
            version=self._infer_version,
            time_range=self._infer_time_range,
            persistence_engine=self._infer_persistence_engine,
            completion_checker=infer_inherited_completion_checker,
        )

    def _infer_version(self, dependencies: t.Mapping[str, Dependency]):
        return self._infer_inherited_param(
            param="version",
            dependencies=dependencies,
            defaults=self.defaults,
            aggregation_method=None,
        )

    def _infer_persistence_engine(self, dependencies: t.Mapping[str, Dependency]):
        return self._infer_inherited_param(
            param="persistence_engine",
            dependencies=dependencies,
            defaults=self.defaults,
            aggregation_method=None,
        )

    def _infer_time_range(self, dependencies: t.Mapping[str, Dependency]):
        return self._infer_inherited_param(
            param="time_range",
            dependencies=dependencies,
            defaults=self.defaults,
            aggregation_method=self._intersect_time_ranges,
        )

    T = t.TypeVar("T")

    def _infer_inherited_param(
        self,
        param,
        dependencies: t.Mapping[str, Dependency],
        defaults: Defaults,
        aggregation_method: t.Optional[t.Callable[[t.AbstractSet[T]], T]],
    ) -> T:

        values = {getattr(dep.task, param) for dep in dependencies.values()}

        default_value = getattr(defaults, param)
        if default_value is not defaults.MISSING:
            values.add(default_value)

        if not values:
            raise ValueError(
                "There are no predecessor or default values from which to infer the "
                f"value of {param}. Please specify this parameter explicitly."
            )

        elif len(values) == 1:
            (value,) = values
            return value

        elif aggregation_method is None:
            raise ValueError(
                f"Multiple predecessor and/or default values found for parameter "
                f"{param} and no aggregation method is defined. Cannot therefore "
                f"disambiguate; please specify this parameter value explicitly in the "
                f"constructor, or update the aggregation method"
            )

        else:
            return aggregation_method(values)

    @staticmethod
    def _intersect_time_ranges(time_ranges: t.Collection[TimeRange]):
        return TimeRange(
            start=max(tr.start for tr in time_ranges),
            end=min(tr.end for tr in time_ranges),
        )
