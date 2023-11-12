from .completion_checking import CalendarChecker, ICompletionChecker, IrregularChecker
from .interface import ICompletionChecker, IStaticTask, ITask, ITimeSeriesTask
from .task import (
    FunctionWrapperMixin,
    StaticFunctionWrapper,
    StaticTaskBase,
    TaskBase,
    TimeSeriesFunctionWrapper,
    TimeSeriesTaskBase,
)
__version__ = "1.0.0"