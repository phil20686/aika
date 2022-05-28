import inspect
from abc import ABC, abstractmethod
from pprint import pformat

import attr
import typing as t

from aika.dux import StaticFunctionWrapper
from aika.dux.concrete import (
    FunctionWrapperMixin,
    task,
    TimeSeriesFunctionWrapper,
    FunctionWrapperUtils,
)
from aika.dux.utils import required_args


class IGraphContextParams(ABC):
    @abstractmethod
    def as_dict(self) -> t.Dict[str, t.Any]:
        pass


ParamsType = t.TypeVar("ParamsType", bound=IGraphContextParams)
FROM_CONTEXT = object()


@attr.s(frozen=True)
class GraphContext(t.Generic[ParamsType]):

    params: ParamsType = attr.ib()

    def _task(
        self,
        function: t.Callable[..., t.Any],
        *args,
        task_cls: t.Type[FunctionWrapperMixin],
        **kwargs,
    ):

        func_sig = inspect.signature(function)
        cls_sig = task_cls.factory_signature()

        if args:
            kwargs = FunctionWrapperUtils.merge_positionals_with_kwargs(
                func_sig=func_sig,
                args=args,
                kwargs=kwargs,
            )

        required = required_args(func_sig) | required_args(cls_sig)
        unspecified = required.difference(kwargs)
        from_context = {
            k: v for k, v in self.params.as_dict().items() if k in unspecified
        }
        missing = unspecified.difference(from_context)

        if missing:
            raise ValueError(
                f"Required arguments {pformat(missing)} were not specified. These must "
                "be specified either explicitly, or included in the context params."
            )

        return task(
            function,
            task_cls,
            **kwargs,
            **from_context,
        )

    def time_series_task(self, function, *args, **kwargs):
        return self._task(function, *args, **kwargs, task_cls=TimeSeriesFunctionWrapper)

    def static_task(self, function, *args, **kwargs):
        return self._task(function, *args, **kwargs, task_cls=StaticFunctionWrapper)
