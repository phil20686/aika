import typing as t

import attr
import numpy as np
import pandas as pd

from aika.utilities.pandas_utils import IndexTensor

from aika.time.timestamp import Timestamp

RESOLUTION = pd.Timedelta(nanoseconds=1)


@attr.s(frozen=True, init=False, repr=False)
class TimeRange:
    """
    Represents a half-open interval (open at the end) between two timestamps.
    Intended to work with ordered pandas time-indexed dataframes, and are valid only
    across the same intervals as pd.Timestamp.
    """

    start = attr.ib()
    end = attr.ib()

    def __init__(
        self,
        start: t.Union[pd.Timestamp, str, None],
        end: t.Union[pd.Timestamp, str, None],
    ):
        if start is None:
            start = pd.Timestamp.min.tz_localize("UTC")
        else:
            start = Timestamp(start)
        if end is None:
            end = pd.Timestamp.max.tz_localize("UTC")
        else:
            end = Timestamp(end)
        if start > end:
            raise ValueError("The start time must be before the end time")

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def view(self, tensor: IndexTensor, level=None) -> IndexTensor:
        """
        Constrains a pandas object to the given time range.

        Parameters
        ----------
        tensor : Tensor
            The pandas object to constrain
        level :
            If multi indexed, the level on which to constrain

        Returns
        -------
        A view of the pandas object constrained to the given time range.
        """
        if isinstance(tensor, pd.Index):
            return self.view(tensor.to_series()).index

        index = tensor.index

        if isinstance(index, pd.MultiIndex):

            if level is None:
                raise ValueError("Must specify `level` if tensor is multi-indexed.")

            level_values = index.get_level_values(level)
            mask = (self.start > level_values) & (level_values <= self.end)
            return tensor.loc[mask]

        else:

            start, end = np.searchsorted(index, [self.start, self.end], side="right")

            return tensor.iloc[start:end]

    @staticmethod
    def from_pandas(tensor: IndexTensor, level=None):
        index = tensor if isinstance(tensor, pd.Index) else tensor.index
        values = index.get_level_values(level)
        return TimeRange(values[0] - RESOLUTION, values[-1])

    def intersects(self, other: "TimeRange") -> bool:
        return (self.start <= other.start < self.end) or (
            other.start <= self.start < other.end
        )

    def contains(self, other: "TimeRange") -> bool:
        """
        Checks whether other is a sub interval of this time range.
        """
        return self.start <= other.start and other.end <= self.end

    def union(self, other: "TimeRange"):
        if not self.intersects(other):
            return TimeRange(pd.NaT, pd.NaT)
        else:
            return TimeRange(
                start=min(self.start, other.start), end=max(self.end, other.end)
            )

    def intersection(self, other: "TimeRange") -> "TimeRange":
        if not self.intersects(other):
            return TimeRange(pd.NaT, pd.NaT)

        else:
            return TimeRange(
                start=max(self.start, other.start), end=min(self.end, other.end)
            )

    def __contains__(self, item):
        return self.start < item <= self.end

    def __repr__(self):
        """
        For ease of use in notebooks it should repr into something that can then
        execute, i.e. it maintains the contract eval(repr(tr)) == tr
        """

        start_str = self._timestamp_repr(self.start)
        end_str = self._timestamp_repr(self.end)

        return f"TimeRange({start_str}, {end_str})"

    @classmethod
    def _timestamp_repr(cls, ts: pd.Timestamp):
        # TODO : Really this belongs on the timestamp class.

        seconds = ts.second + ts.microsecond / 1e6 + ts.nanosecond / 1e9
        if seconds < 10.0:
            seconds_repr = f"0{seconds}"
        else:
            seconds_repr = str(seconds)
        # this works because we are guaranteed olson timezones.
        str_format = "'%Y-%m-%dT%H:%M:{seconds} [{tz}]'"
        return ts.strftime(str_format).format(seconds=seconds_repr, tz=ts.tz)
