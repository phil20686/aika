from ._version import __version__

from .interface import ICompletionChecker, ITask, ITimeSeriesTask, IStaticTask
from .completion_checking import ICompletionChecker, IrregularChecker, CalendarChecker
from .concrete import (
    TaskBase,
    TimeSeriesTaskBase,
    StaticTaskBase,
    FunctionWrapperMixin,
    TimeSeriesFunctionWrapper,
    StaticFunctionWrapper,
    time_series_task,
    static_task,
)
