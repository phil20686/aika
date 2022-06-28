from pprint import pprint

import pandas as pd

from aika import putki
from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine
from aika.time.calendars import TimeOfDayCalendar
from aika.time.time_of_day import TimeOfDay
from aika.time.time_range import TimeRange

from aika.putki import CalendarChecker
from aika.putki.context import Defaults, GraphContext
from aika.putki.graph import Graph, TaskModule
from aika.putki.runners import LocalRunner


def generate_index(time_range: TimeRange, time_of_day: TimeOfDay):

    return TimeOfDayCalendar(time_of_day=time_of_day).to_index(time_range)


def compound_interest(calendar: pd.Index, interest_rate: float):
    return pd.Series(1 + interest_rate, index=calendar).cumprod()


class TimesOfDay:

    LONDON_1420 = TimeOfDay.from_str("14:20 [Europe/London]")


class ReferenceDataTasks(TaskModule):
    def __init__(self, context: GraphContext):

        self.CALENDAR = context.time_series_task(
            "CALENDAR",
            generate_index,
            time_of_day=TimesOfDay.LONDON_1420,
            completion_checker=CalendarChecker(
                TimeOfDayCalendar(TimesOfDay.LONDON_1420)
            ),
        )


class MarketDataTasks(TaskModule):
    def __init__(
        self,
        context: GraphContext,
        reference_data: ReferenceDataTasks,
    ):

        self.RETURNS = context.time_series_task(
            "RETURNS",
            compound_interest,
            interest_rate=0.01,
            calendar=reference_data.CALENDAR,
        )


class AllTasks(TaskModule):
    def __init__(self, context: GraphContext):

        self.reference_data = ReferenceDataTasks(
            context.extend_namespace("reference_data")
        )

        self.market_data = MarketDataTasks(
            context=context.extend_namespace("market_data"),
            reference_data=self.reference_data,
        )


defaults = Defaults(
    version=putki.__version__,
    persistence_engine=HashBackedPersistanceEngine(),
    time_range=TimeRange("2021", "2022"),
)

ctx = GraphContext(defaults=defaults)

tasks = AllTasks(ctx)
graph = Graph(tasks.all_tasks)
runner = LocalRunner()
runner.run(graph)

pprint(tasks.all_tasks)

assert tasks.reference_data.CALENDAR.complete()
assert tasks.market_data.RETURNS.complete()

print(tasks.market_data.RETURNS.read())
