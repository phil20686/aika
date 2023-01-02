import logging
from multiprocessing import cpu_count

try:
    import luigi

    luigi_available = True
    logging.getLogger(__name__).info("Luigi not available, creating luigi runner")
except ModuleNotFoundError:
    luigi_available = False
    logging.getLogger(__name__).info("Luigi not available, skipping luigi runner")

if luigi_available:
    try:
        from functools import cached_property
    except ImportError:
        # if python version < 3.8.
        from backports.cached_property import cached_property
    from numbers import Number
    import pandas as pd
    from luigi import (
        IntParameter,
        FloatParameter,
        DictParameter,
        ListParameter,
        BoolParameter,
        Parameter,
        Target,
        OptionalParameter,
    )
    from typing import Mapping, Sequence

    from aika.putki import ITask
    from aika.putki.graph import Graph
    from aika.putki.runners import IGraphRunner, GraphStatus
    from aika.time import TimeRange, Timestamp

    def _list_to_tuple(x):
        """Make tuples out of lists and sets to allow hashing"""
        if isinstance(x, list) or isinstance(x, set):
            return tuple(x)
        else:
            return x

    class AikaTarget(Target):
        def __init__(self, aika_task: ITask):
            self._aika_task = aika_task

        def exists(self):
            return self._aika_task.complete()

    class Parameters:
        class TimeRangeParameter(luigi.Parameter):
            def parse(self, s: str):
                return TimeRange.from_string(s)

            def serialize(self, x: TimeRange):
                return repr(x)

        class Timestamp(luigi.Parameter):
            def parse(self, s: str):
                return Timestamp(s)

            def serialize(self, x: pd.Timestamp):
                return TimeRange._timestamp_repr(x)

    class LuigiTaskFromTask(luigi.Task):
        def get_param_values(self, params, args, kwargs):
            # note that params are assumed hashable
            param_values = {
                **dict(
                    name=self._metadata.name,
                    version=self._metadata.version,
                    static=self._metadata.static,
                ),
                **self._metadata.params,
                **(
                    dict(
                        time_level=self._aika_task.time_level,
                        time_range=self._aika_task.time_range,
                    )
                    if self._aika_task.time_series
                    else {}
                ),
            }
            return [
                (param_name, _list_to_tuple(param_values[param_name]))
                for param_name in param_values.keys()
            ]

        def get_params(self):
            params = []
            for name, value in self.get_param_values(None, None, None):
                if isinstance(value, bool):
                    params.append(
                        (
                            name,
                            BoolParameter(
                                significant=True,
                                description="Bool parameter dynamically created by aika",
                                positional=False,
                            ),
                        )
                    )
                elif isinstance(value, int):
                    params.append(
                        (
                            name,
                            IntParameter(
                                significant=True,
                                description="Int parameter dynamically created by aika",
                                positional=False,
                            ),
                        )
                    )
                elif isinstance(value, str):
                    params.append(
                        (
                            name,
                            Parameter(
                                significant=True,
                                description="String parameter dynamically created by aika",
                                positional=False,
                            ),
                        )
                    )
                elif isinstance(value, pd.Timestamp):
                    params.append(
                        (
                            name,
                            Parameters.Timestamp(
                                significant=True,
                                description="Timestamp parameter dynamically created by aika",
                                positional=False,
                            ),
                        )
                    )
                elif isinstance(value, TimeRange):
                    params.append(
                        (
                            name,
                            Parameters.TimeRangeParameter(
                                significant=True,
                                description="Time range parameter dynamically created by aika",
                                positional=False,
                            ),
                        )
                    )
                elif isinstance(value, Number):
                    params.append(
                        (
                            name,
                            FloatParameter(
                                significant=True,
                                description="Float parameter dynamically created by aika",
                                positional=False,
                            ),
                        )
                    )
                elif isinstance(value, Mapping):
                    params.append(
                        (
                            name,
                            DictParameter(
                                significant=True,
                                description="Dict parameter dynamically created by aika",
                                positional=False,
                            ),
                        )
                    )
                elif isinstance(value, Sequence):
                    params.append(
                        (
                            name,
                            ListParameter(
                                significant=True,
                                description="List parameter dynamically created by aika",
                                positional=False,
                            ),
                        )
                    )
                elif value is None:
                    params.append(
                        (
                            name,
                            OptionalParameter(
                                significant=True,
                                description="optional string parameter created for null value",
                                positional=False,
                            ),
                        )
                    )
                else:
                    raise ValueError(
                        f"Unable to dynamically create a parameter for {name} and {value}"
                    )
            return params

        @classmethod
        def batch_param_names(cls):
            return []

        def complete(self):
            return self._aika_task.complete()

        def clone(self, cls=None, **kwargs):
            # just create new aika tasks!
            raise NotImplementedError

        def output(self):
            return [AikaTarget(self._aika_task)]

        def requires(self):
            return [
                LuigiTaskFromTask(dep.task)
                for dep in self._aika_task.dependencies.values()
            ]

        def from_str_params(cls, params_str):
            raise ValueError("I do not think this should be needed")

        def __init__(self, aika_task: ITask):
            self._aika_task = aika_task
            self._metadata = self._aika_task.output

            full_param_values = self.get_param_values(None, None, None)
            # Set all values on class instance
            for key, value in full_param_values:
                setattr(self, key, value)
            self.param_kwargs = dict(full_param_values)

            self._warn_on_wrong_param_types()
            self.task_id = f"aika_{self._metadata.name}_{self._metadata.version}_{hash(self._metadata)}"
            # self.__hash = self._aika_task.__hash__()

            self.set_tracking_url = None
            self.set_status_message = None
            self.set_progress_percentage = None

        def __hash__(self):
            return self._aika_task.__hash__()

        def run(self):
            return self._aika_task.run()

    LuigiTaskFromTask.disable_instance_cache()

    class LuigiRunner(IGraphRunner):
        """
        Luigi runner has a number of noteable weaknesses around multiprocessing when run on window,
        see the luigi docs for details. The most common luigi kwargs that you might use are explicit,
        but the **kwargs will pass any more kwargs into the luigi.build command. Note that if
        use_local_scheduler is False, then you should seperately have started up a luigi scheduler,
        again see the luigi docs for details.

        Luigi is not currently available for 3.11.0 as of writing, so this will not work in 3.11 hence
        the if statements at the top of this class.
        """

        def __init__(
                self,
                scheduler_host="localhost",
                scheduler_port=8082,
                scheduler_url=None,
                use_local_scheduler=False,
                workers=max(cpu_count()-1, 1),
                **more_luigi_kwargs
        ):
            self._scheduler_host = scheduler_host
            self._scheduler_port = scheduler_port
            self._scheduler_url = scheduler_url or f"http://{self._scheduler_host}:{self._scheduler_port}"
            self._use_local_scheduler = use_local_scheduler
            self._workers = workers
            self._more_luigi_kwargs = more_luigi_kwargs


        def run(self, graph: Graph):
            original_status = GraphStatus(graph)
            sinks = graph.sinks
            luigi_summary = luigi.build(
                [LuigiTaskFromTask(task) for task in sinks],
                workers=self._workers,
                local_scheduler=self._use_local_scheduler,
                scheduler_host=self._scheduler_host,
                scheduler_url=self._scheduler_url,
                scheduler_port=self._scheduler_port,
                detailed_summary=True,
                **self._more_luigi_kwargs
            )
            logging.getLogger(__name__).info(
                f"Luigi Summary\n {luigi_summary.summary_text}"
            )

            # this is all a bit wasteful but it basically goes out
            # and checks the status after luigi has finished. Completion checking
            # is quite fast so should be a minimal performance overhead.
            ready = set()
            ready.update(original_status.ready)
            while len(ready):
                task = ready.pop()
                if task.complete():
                    ready.update(original_status.assert_ran_successfully(task))
                else:
                    original_status.assert_permanent_failure(task)
            return original_status
