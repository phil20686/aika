import typing as t
from calendar import MONDAY

import numpy as np
from abc import ABC, abstractmethod

import attr
import pandas as pd
from pandas._libs.tslibs.offsets import BDay, BusinessDay

from ebony.datagraph.interface import DataSetMetadata, IPersistenceEngine
from ebony.time.calendars import ICalendar
from ebony.time.time_range import TimeRange, RESOLUTION


class ICompletionChecker(ABC):
    def is_complete(
        self,
        metadata: "DataSetMetadata",
        engine: "IPersistenceEngine",
        target_time_range: TimeRange,
    ) -> bool:
        pass


@attr.s(frozen=True)
class CalendarChecker(ICompletionChecker):

    calendar: ICalendar = attr.ib()

    def is_complete(
        self,
        metadata: "DataSetMetadata",
        engine: "IPersistenceEngine",
        target_time_range: TimeRange,
    ) -> bool:
        required_time_range = TimeRange(
            start=self.calendar.latest_point_before(target_time_range.end),
            end=target_time_range.end,
        )
        actual_time_range = engine.get_data_time_range(metadata)
        return required_time_range.intersects(actual_time_range)


@attr.s(frozen=True)
class IrregularChecker(ICompletionChecker):
    def is_complete(
        self,
        metadata: "DataSetMetadata",
        engine: "IPersistenceEngine",
        target_time_range: TimeRange,
    ) -> bool:
        declared_time_range = engine.get_declared_time_range(metadata)
        return declared_time_range.end >= target_time_range.end
