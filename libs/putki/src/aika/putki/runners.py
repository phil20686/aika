import collections
import copy
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import Iterable, Set

from aika.putki import ITask
from aika.putki.graph import Graph


class GraphStatus:
    """
    This helps keep track of a graph that is running. Think of it as state machine that you
    can tell which tasks have completed or failed.
    """

    @property
    def complete(self) -> Set[ITask]:
        """
        Tasks which were checked and are complete. Note that when a graph status is initialised
        it will try to find the smallest possible graph that needs to run, so in a completed
        graph len(complete) < len(self._graph) is possible.
        """
        return copy.copy(self._complete)

    @property
    def failed(self) -> Set[ITask]:
        """
        Tasks that were attempted to run and which failed.
        """
        return copy.copy(self._failed)

    @property
    def ready(self) -> Set[ITask]:
        """
        Tasks that are currently ready to run - i.e. their parents have completed.
        """
        return copy.copy(self._ready_to_run)

    @property
    def waiting(self) -> Set[ITask]:
        """
        Tasks that have at least one predecessor that is not yet complete.
        """
        return copy.copy(self._need_to_run.difference(self._ready_to_run))

    @property
    def has_failed_predecessor(self) -> Set[ITask]:
        """
        Tasks that have at least one predecessor that has failed.
        """
        return copy.copy(self._failed_predecessors)

    def __str__(self):
        return f"""
Graph Status:
    Graph contains {len(self._graph)} tasks
    {len(self.complete)} are complete
    {len(self.failed)} have failed
    {len(self.ready)} are ready to run
    {len(self.waiting)} are waiting for predecessors to complete
    {len(self.has_failed_predecessor)} cannot be run because a predecessor failed.
    The tasks which directly failed are:
        """ + "\n\t\t".join(
            sorted([t.name for t in self.failed])
        )

    def __init__(self, graph: Graph):
        self._complete = set()
        self._need_to_run = set()
        self._failed = set()
        self._failed_predecessors = set()
        self._graph = graph
        self._ready_to_run = set()

        stack = collections.deque()
        stack.extend(graph.sinks)
        visited = set()
        while stack:
            task = stack.popleft()
            if task not in visited:
                visited.add(task)
                if task.complete():
                    self._complete.add(task)
                else:
                    self._need_to_run.add(task)
                    for dep in task.dependencies.values():
                        if dep.task not in visited:
                            stack.append(dep.task)

        self._initialise_ready_to_run()

    def _task_dependencies(self, task: ITask) -> Iterable[ITask]:
        """
        Helper method to get the dependencies as tasks
        """
        return [d.task for d in task.dependencies.values()]

    def _all_dependencies_complete(self, task: ITask) -> bool:
        return all([(dep in self._complete) for dep in self._task_dependencies(task)])

    def _initialise_ready_to_run(self):
        for task in self._need_to_run:
            if self._all_dependencies_complete(task):
                self._ready_to_run.add(task)

    def assert_ran_successfully(self, task: ITask) -> Set[ITask]:
        """
        Tell the status object that a task has completed. This will then return
        a set of tasks which are now able to be run. These will be added internally
        to the ready property, but for ease they are also returned.
        """
        self._need_to_run.discard(task)
        self._ready_to_run.discard(task)
        self._complete.add(task)
        new_additions = set()
        for successor in self._graph.get_successors(task):
            if self._all_dependencies_complete(successor):
                new_additions.add(successor)
        self._ready_to_run.update(new_additions)
        return new_additions

    def assert_permanent_failure(self, task: ITask) -> Set[ITask]:
        """
        Tell the status object that a task has failed and should not be re-run. This will then
        remove any successors of that task from the set of tasks waiting to be run. It will return
        any new additions to the has_failed_predecessor property.
        """
        self._need_to_run.discard(task)
        self._ready_to_run.discard(task)
        self._failed.add(task)

        stack = collections.deque()
        stack.extend(self._graph.get_successors(task))
        new_fails = set()
        while stack:
            successor = stack.popleft()
            self._need_to_run.discard(successor)
            self._failed_predecessors.add(successor)
            new_fails.add(successor)
            for grand_successor in self._graph.get_successors(successor):
                if grand_successor not in self._failed_predecessors:
                    stack.append(grand_successor)
        self._failed_predecessors.update(new_fails)
        return new_fails


class IGraphRunner(ABC):
    @abstractmethod
    def run(self, graph: Graph) -> GraphStatus:
        pass


class SingleThreadedRunner(IGraphRunner):
    """
    This will run all the tasks in a single thread. Created as a reference implementation
    and also to aid debugging.
    """

    @classmethod
    def run(cls, graph: Graph) -> GraphStatus:
        status = GraphStatus(graph)

        stack = collections.deque()
        stack.extend(status.ready)

        while stack:
            task = stack.popleft()
            try:
                task.run()
                stack.extend(status.assert_ran_successfully(task))
            except KeyboardInterrupt as e:
                raise e  # pragma: no cover
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Task {task.name} failed to run successfully"
                )
                status.assert_permanent_failure(task)
        return status


class MultiThreadedRunner(IGraphRunner):
    """
    Multi-threaded runner. This will use multiprocess to farm out the task objects to multiple threads.
    Since the scheduler controls the order in which tasks are operated, it is safe to use such a scheduler
    with nominally non-thread safe persistence engines, since no single task will ever be being run simultaneously
    by two threads. It is however, not safe to use with an in memory persistence engine.
    """

    @classmethod
    def run(cls, graph: Graph, max_threads=cpu_count()) -> GraphStatus:
        executor = ProcessPoolExecutor(max_workers=max_threads)
        status = GraphStatus(graph)

        futures = {}
        ready = set()
        ready.update(status.ready)

        while len(futures) + len(ready):
            while ready:
                task = ready.pop()
                new_future = executor.submit(task.run)
                futures[task] = new_future
            for t, f in copy.copy(futures).items():
                if not f.done():
                    continue
                else:
                    if f.cancelled() or f.exception(0):
                        status.assert_permanent_failure(t)
                        futures.pop(t)
                    else:
                        newly_ready = status.assert_ran_successfully(t)
                        futures.pop(t)
                        ready.discard(t)
                        ready.update(newly_ready)
            time.sleep(0.01)
        return status
