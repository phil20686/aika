import typing as t

try:
    from functools import cached_property
except ImportError:
    # if python version < 3.8.
    from backports.cached_property import cached_property


import networkx as nx

from aika.putki.interface import ITask


class TaskModule:
    @cached_property
    def all_tasks(self) -> t.AbstractSet[ITask]:  # pragma: no cover
        result = set()
        for value in self.__dict__.values():
            if isinstance(value, ITask):
                result.add(value)
            elif isinstance(value, TaskModule):
                result.update(value.all_tasks)

        return result


class Graph:
    def __len__(self):
        return len(self._nodes)

    def __init__(self, tasks: t.Collection[ITask]):
        self._nodes = set()
        self._edges = set()
        frontier = set(tasks)

        while frontier:
            task = frontier.pop()
            if task not in self._nodes:
                self._nodes.add(task)

                for dep in task.dependencies.values():
                    if dep.task not in self._nodes:
                        frontier.add(dep.task)

                    self._edges.add((dep.task, task))

        self.graph = nx.DiGraph(self._edges)

    def get_successors(self, task: ITask) -> t.Iterator[ITask]:
        try:
            return self.graph.successors(task)
        except nx.NetworkXError as e:
            raise ValueError(
                f"Requested the successor of a task {task.name} that is not in the graph"
            )

    @property
    def all_tasks(self) -> t.Set[ITask]:  # pragma: no cover
        return self._nodes

    @cached_property
    def sinks(self) -> t.AbstractSet[ITask]:
        return frozenset(
            node
            for node in self._nodes
            if (
                node not in self.graph.nodes
                or len(list(self.graph.successors(node))) == 0
            )
        )
