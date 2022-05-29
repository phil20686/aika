from pprint import pprint

import attr
import pandas as pd

from aika import dux
from aika.datagraph.interface import IPersistenceEngine
from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
from aika.dux import CalendarChecker
from aika.dux.graph import GraphContext, IGraphContextParams, TaskModule, Graph
from aika.dux.runners import LocalRunner
from aika.time.calendars import TimeOfDayCalendar
from aika.time.time_of_day import TimeOfDay
from aika.time.time_range import TimeRange


def generate_index(time_range: TimeRange, time_of_day: TimeOfDay):

    return TimeOfDayCalendar(time_of_day=time_of_day).to_index(time_range)


def compound_interest(calendar: pd.Index, interest_rate: float):
    return pd.Series(1 + interest_rate, index=calendar).cumprod()


@attr.s(auto_attribs=True)
class Params(IGraphContextParams):

    namespace: str
    version: str
    persistence_engine: IPersistenceEngine
    time_range: TimeRange

    def as_dict(self):
        return attr.asdict(self, recurse=False)

    def update(self, **overrides):
        return attr.evolve(self, **overrides)


class TimesOfDay:

    LONDON_1420 = TimeOfDay.from_str("14:20 [Europe/London]")


class ReferenceDataTasks(TaskModule):
    def __init__(self, context: GraphContext[Params]):

        self.context = context.update(namespace="reference_data")

        self.CALENDAR = self.context.time_series_task(
            generate_index,
            name="CALENDAR",
            time_of_day=TimesOfDay.LONDON_1420,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(TimesOfDay.LONDON_1420)
            ),
        )


class MarketDataTasks(TaskModule):
    def __init__(
        self, context: GraphContext[Params], reference_data: ReferenceDataTasks
    ):

        self.context = context.update(namespace="market_data")

        self.RETURNS = self.context.time_series_task(
            compound_interest,
            name="RETURNS",
            interest_rate=0.01,
            calendar=reference_data.CALENDAR,
        )


class AllTasks(TaskModule):
    def __init__(self, context: GraphContext[Params]):

        self.reference_data = ReferenceDataTasks(context)
        self.market_data = MarketDataTasks(context, reference_data=self.reference_data)


context = GraphContext(
    params=Params(
        namespace="default_namespace",
        version=dux.__version__,
        persistence_engine=HashBackedPersistanceEngine(),
        time_range=TimeRange("2021", "2022"),
    )
)

tasks = AllTasks(context)
graph = Graph(tasks.all_tasks)
runner = LocalRunner()
runner.run(graph)

pprint(tasks.all_tasks)

assert tasks.reference_data.CALENDAR.complete()
assert tasks.market_data.RETURNS.complete()

print(tasks.market_data.RETURNS.read())
