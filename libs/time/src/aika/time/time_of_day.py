import datetime
import re
import typing as t
from abc import abstractmethod

import attr
import pandas as pd
import pytz
from typing_extensions import Protocol


class _DateLike(Protocol):
    @property
    @abstractmethod
    def day(self) -> int:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def month(self) -> int:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def year(self) -> int:
        pass  # pragma: no cover


def _parse_timezone(tz_or_str: t.Union[datetime.tzinfo, str]) -> datetime.tzinfo:
    if isinstance(tz_or_str, str):
        return pytz.timezone(tz_or_str)
    else:
        return tz_or_str


@attr.s(frozen=True, repr=False)
class TimeOfDay:

    time: datetime.time = attr.ib()
    tz: t.Union[datetime.tzinfo] = attr.ib(converter=_parse_timezone)

    def __repr__(self):
        return f"{self.time.strftime('%H:%M:%S.%f')} [{str(self.tz)}]"

    @classmethod
    def from_str(cls, s):

        pattern = r"^([^\[\s]*)\s*(?:\[(.*)\])?$"

        match = re.match(pattern, s)
        if not match:
            raise ValueError(f"Failed to parse string {s} as a TimeOfDay")

        time_str, tz_str = match.groups()

        formats = [
            "%H:%M:%S.%f",
            "%H:%M:%S",
            "%H:%M",
        ]

        for fmt in formats:
            try:
                time = datetime.datetime.strptime(time_str, fmt).time()
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                f"Time string {time_str} did not match any of the formats {formats}"
            )

        tz = pytz.timezone(tz_str) if tz_str is not None else pytz.UTC

        return cls(time=time, tz=tz)

    def make_timestamp(self, date: _DateLike) -> pd.Timestamp:
        return pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=self.time.hour,
            minute=self.time.minute,
            second=self.time.second,
            microsecond=self.time.microsecond,
            tz=self.tz,
        )
