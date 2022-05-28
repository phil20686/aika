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

    # it is assumed that these will be constructor arguments in concrete subclasses
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

    @classmethod
    def factory_signature(cls):
        """
        The `factory` method below depends on inspecting the constructor signature of
        `cls` to parse the correct keyword arguments to pass to the class constructor.

        However, in addition to this dynamically-inferred kwargs dict, the factory also
        explicitly passes in `function`, `scalar_func_kwargs`, and `func_dependencies`
        as top-level arguments. As a result, it is useful to be able to reference the
        constructor signature of `cls` _without_ those arguments. This classmethod
        returns precisely that.
        """
        full_signature = inspect.signature(cls)

        private_params = {
            "function",
            "scalar_func_kwargs",
            "func_dependencies",
        }

        remaining_params = [
            param
            for param in full_signature.parameters.values()
            if param.name not in private_params
        ]

        return full_signature.replace(parameters=remaining_params)

    @classmethod
    def factory(cls, function, /, *args, **kwargs):
        """
        This factory method takes a function and a combination of function arguments and
        class constructor arguments, and returns an instance of `cls` with the
        function-specific arguments reformatted appropriately into
        `scalar_func_kwargs` and `func_dependencies` appropriately.

        Accepts positional args, which are always interpreted as function arguments.

        In the event that `function` and `cls` share any arguments, the value specified
        for that argument will be passed to both.

        Any dependency specified as a pure ITask instance rather than a Dependency
        object will also be automatically wrapped into a Dependency.

        Note that this factory relies on the assumption that the three abstract
        attributes defined on `FunctionWrapperMixin` (namely: `function`,
        `scalar_func_kwargs`, and `func_dependencies`) are all constructor arguments
        on the `cls`.
        """

        func_sig = inspect.signature(function)
        cls_sig = cls.factory_signature()

        FunctionWrapperUtils.validate_param_kinds(
            signature=func_sig, name=function.__qualname__
        )

        if args:
            kwargs = FunctionWrapperUtils.merge_positionals_with_kwargs(
                func_sig,
                args=args,
                kwargs=kwargs,
            )

        func_kwargs, cls_kwargs = FunctionWrapperUtils.split_func_and_cls_kwargs(
            func_sig=func_sig,
            func_name=function.__qualname__,
            cls_sig=cls_sig,
            cls_name=cls.__name__,
            kwargs=kwargs,
        )

        (
            scalar_func_kwargs,
            func_dependencies,
        ) = FunctionWrapperUtils.split_scalars_and_dependencies(func_kwargs)

        # noinspection PyArgumentList
        return cls(
            function=function,
            scalar_func_kwargs=frozendict(scalar_func_kwargs),
            func_dependencies=frozendict(func_dependencies),
            **cls_kwargs,
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


time_series_task = TimeSeriesFunctionWrapper.factory
static_task = StaticFunctionWrapper.factory


class FunctionWrapperUtils:
    """
    This class contains a collection of utility functions used by the `task` factory
    function to validate, parse, and reformat its inputs.
    """

    @classmethod
    def validate_param_kinds(cls, signature: inspect.Signature, name: str):
        """
        Verify that the provided function signature contains only named arguments which
        can be passed via keyword arguments - i.e. no positional-only arguments, and no
        *args or **kwargs catch-alls. This ensures that the signature can be accurately
        parsed and validated.
        """

        bad_params = {
            inspect.Parameter.POSITIONAL_ONLY: [],
            inspect.Parameter.VAR_POSITIONAL: [],
            inspect.Parameter.VAR_KEYWORD: [],
        }
        any_bad = False

        for name, param in signature.parameters.items():
            if param.kind in bad_params:
                bad_params[param.kind].append(name)
                any_bad = True

        if any_bad:
            raise ValueError(
                "function must not have any positional-only, var-positional "
                "(e.g. *args), or var-keyword (e.g. **kwargs) parameters in its "
                f"signature. Specified function {name} has the following invalid "
                f"parameters in its signature: {pformat(bad_params)}"
            )

    @classmethod
    def merge_positionals_with_kwargs(
        cls,
        func_sig: inspect.Signature,
        args: t.Sequence[t.Any],
        kwargs: t.Dict[str, t.Any],
    ):
        """
        Given a sequence `args` of positional arguments, interpret these according to
        the specified function signature in order to bind them to parameter names.

        Also validates that after binding the args to names, there are no collisions
        with `kwargs`.

        Note that `args` must be a valid argument sequence for `func_sig`, but no such
        assumption is applied to `kwargs` - in practice, this function is called with
        `kwargs` dicts which include both function and task arguments (see the `task`
        function above).
        """

        kwargs_from_positionals = func_sig.bind_partial(*args).arguments

        if duplicated := set(kwargs_from_positionals).intersection(kwargs):
            positional_values = {k: kwargs_from_positionals[k] for k in duplicated}
            keyword_values = {k: kwargs[k] for k in duplicated}

            raise ValueError(
                f"Parameters {duplicated} were specified as both positional and "
                f"keyword arguments. The positional values were "
                f"{pformat(positional_values)}, while the keyword values where "
                f"{pformat(keyword_values)}. Resolve this ambiguity by specifying "
                f"each argument positionally or as a keyword, not both."
            )

        # as we passed the above error check, this is guaranteed to be a disjoint union
        return kwargs_from_positionals | kwargs

    @classmethod
    def split_func_and_cls_kwargs(
        cls,
        func_sig: inspect.Signature,
        func_name: str,
        cls_sig: inspect.Signature,
        cls_name: str,
        kwargs: t.Dict[str, t.Any],
    ) -> t.Tuple[t.Dict[str, t.Any], t.Dict[str, t.Any]]:
        """
        Given a single dict `kwargs` of keyword arguments, and two signatures for a
        function and task class respectively, split `kwargs` into two dicts, containing
        the arguments appropriate for each respective signature.

        In the event that an argument appears in both signatures, it will be returned
        in both dicts.

        In the event that an element of `kwargs` appears in neither signature, an error
        will be raised.

        The two output dicts are returned as a pair, (func_kwargs, cls_kwargs).
        """

        expected_kwargs = set(func_sig.parameters) | set(cls_sig.parameters)

        unexpected = {}
        func_kwargs = {}
        cls_kwargs = {}

        for k, v in kwargs.items():
            if k not in expected_kwargs:
                unexpected[k] = v
                continue

            if k in func_sig.parameters:
                func_kwargs[k] = v

            if k in cls_sig.parameters:
                cls_kwargs[k] = v

        if unexpected:
            raise ValueError(
                f"Received values for unexpected parameters: {pformat(unexpected)}. "
                f"These parameters appear neither in the signature of the function "
                f"{func_name}, nor the task class {cls_name}."
            )

        return func_kwargs, cls_kwargs

    @classmethod
    def split_scalars_and_dependencies(cls, func_kwargs: t.Dict[str, t.Any]):
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
