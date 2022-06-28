import inspect
import typing as t
from abc import ABC, abstractmethod
from functools import cached_property
from pprint import pformat

import attr
import networkx as nx

from aika.putki.interface import ITask


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
