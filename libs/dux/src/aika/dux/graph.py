import inspect
from abc import ABC, abstractmethod
from functools import cached_property
from pprint import pformat

import attr
import typing as t

import networkx as nx

from aika.dux import StaticFunctionWrapper, ITask
from aika.dux.concrete import (
    FunctionWrapperMixin,
    TimeSeriesFunctionWrapper,
    FunctionWrapperUtils,
)
from aika.dux.utils import required_args


class IGraphContextParams(ABC):
    @abstractmethod
    def as_dict(self) -> t.Dict[str, t.Any]:
        pass

    @abstractmethod
    def update(self, **overrides) -> "IGraphContextParams":
        pass


P = t.TypeVar("P", bound=IGraphContextParams)


@attr.s(frozen=True)
class GraphContext(t.Generic[P]):

    params: P = attr.ib()

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

        return task_cls.factory(
            function,
            **kwargs,
            **from_context,
        )

    def time_series_task(self, function, *args, **kwargs):
        return self._task(function, *args, **kwargs, task_cls=TimeSeriesFunctionWrapper)

    def static_task(self, function, *args, **kwargs):
        return self._task(function, *args, **kwargs, task_cls=StaticFunctionWrapper)

    def update(self, **overrides) -> "GraphContext[P]":
        return GraphContext(params=self.params.update(**overrides))


class TaskModule:
    @cached_property
    def all_tasks(self) -> t.AbstractSet[ITask]:

        result = set()

        for value in self.__dict__.values():
            if isinstance(value, ITask):
                result.add(value)
            elif isinstance(value, TaskModule):
                result.update(value.all_tasks)

        return result


class Graph:
    def __init__(self, tasks: t.Collection[ITask]):
        nodes = set()
        edges = set()
        frontier = set(tasks)

        while frontier:
            task = frontier.pop()
            nodes.add(task)

            for dep in task.dependencies.values():
                if dep.task not in nodes:
                    frontier.add(dep.task)

                edges.add((dep.task, task))

        self.graph = nx.DiGraph(edges)

    @cached_property
    def sinks(self) -> t.AbstractSet[ITask]:
        return frozenset(
            node for node in self.graph.nodes if not list(self.graph.successors(node))
        )

    def run(self):
        for task in self.sinks:
            task.run_recursively()
