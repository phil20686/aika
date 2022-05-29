from abc import ABC, abstractmethod

from aika.dux.graph import Graph


class IGraphRunner(ABC):
    @abstractmethod
    def run(self, graph: Graph):
        pass


class LocalRunner(IGraphRunner):
    @classmethod
    def _run_task(cls, task):
        if task.complete():
            return

        for dep in task.dependencies.values():
            cls._run_task(dep.task)

        task.run()

    @classmethod
    def run(cls, graph: Graph):
        for task in graph.sinks:
            cls._run_task(task)


class LuigiRunner(IGraphRunner):

    # TODO
    pass
