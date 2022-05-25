import numpy as np
import pandas as pd

from aika.time.calendars import OffsetCalendar
from aika.time.time_range import TimeRange
from aika.utilities.pandas_utils import Tensor

from aika.dux.completion_checking import CalendarChecker
from aika.dux.implementation import ResearchRunner


def _some_random_data(seed: int, time_range: TimeRange):
    index = pd.date_range(start=time_range.start, end=time_range.end, freq="B")
    return pd.DataFrame(
        data=np.random.RandomState(seed=seed).randn(len(index), 5), index=index
    )


def _moving_average(data: Tensor, window: int) -> Tensor:
    return data.rolling(window=window, min_periods=window).mean()


runner = ResearchRunner(
    time_range=TimeRange("2020-01-01", "2021-01-01"),
    lookback=pd.offsets.BDay(n=5),
    autorun=True,
)

task1 = runner.add_timeseries_task(
    "random",
    _some_random_data,
    seed=42,
    completion_checker=CalendarChecker(OffsetCalendar("24H")),
)

task2 = runner.add_timeseries_task(
    "moving_average",
    _moving_average,
    data=task1,
    window=20,
)

task3 = runner.add_timeseries_task(
    "moving_average", _moving_average, data=task1, window=10
)

print(task2.read().tail())
print(task3.read().tail())
