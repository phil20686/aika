import numpy as np
import pandas as pd

from aika.time.calendars import OffsetCalendar
from aika.time.time_range import TimeRange
from aika.utilities.pandas_utils import Tensor

from aika.dux.completion_checking import CalendarChecker
from aika.dux.implementation import TimeSeriesTask


def _some_random_data(seed: int, time_range: TimeRange):
    index = pd.date_range(start=time_range.start, end=time_range.end, freq="B")
    return pd.DataFrame(
        data=np.random.RandomState(seed=seed).randn(len(index), 5), index=index
    )


def _moving_average(data: Tensor, window: int) -> Tensor:
    return data.rolling(window=window, min_periods=window).mean()


from aika.datagraph.persistence.hash_backed import HashBackedPersistanceEngine

engine = HashBackedPersistanceEngine()

tr = TimeRange("2020-01-01", "2021-01-01")
leaf_task = TimeSeriesTask(
    func=_some_random_data,
    time_range=tr,
    completion_checker=CalendarChecker(OffsetCalendar("24H")),
    name="random",
    persistence_engine=engine,
    seed=42,
)

leaf_task.run()

child_task = TimeSeriesTask(
    func=_moving_average,
    data=leaf_task,
    time_range=TimeRange("2020-06-01", "2021-01-01"),
    completion_checker=None,
    name="moving_average",
    persistence_engine=engine,
    default_lookback=pd.offsets.BDay(n=5),
    window=20,
)

child_task.run()

print(engine.read(child_task.output))
