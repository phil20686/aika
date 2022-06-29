import re
import typing as t

import attr
import numpy as np
import pandas as pd

from aika.utilities.pandas_utils import IndexTensor

from aika.time.timestamp import Timestamp
from aika.time.utilities import _get_index_level

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
        if start >= end:
            raise ValueError(
                f"The start time {start} must be before the end time {end}"
            )

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
            return self.view(tensor.to_series(), level=level).index

        index = tensor.index

        if isinstance(index, pd.MultiIndex):

            if level is None:
                raise ValueError("Must specify `level` if tensor is multi-indexed.")

            level_values = index.get_level_values(level)
            mask = (self.start <= level_values) & (level_values < self.end)
            return tensor.loc[mask]

        else:

            start, end = np.searchsorted(index, [self.start, self.end], side="left")

            return tensor.iloc[start:end]

    @staticmethod
    def from_pandas(tensor: IndexTensor, level=None):
        """
        Extracts a time range from the index (level) of a given pandas object.

        Notes
        -----

        Index must be ordered in the given level.

        Returns
        -------
        TimeRange: a time range such that the "view" would extract this exact range from
        a larger dataframe. I.e. the same as a time range (min, max+RESOLUTION).
        """
        values = _get_index_level(tensor, level)
        if values.empty:
            raise ValueError("Cannot extract time range from empty index")
        else:
            return TimeRange(values[0], values[-1] + RESOLUTION)

    def intersects(self, other: "TimeRange") -> bool:
        if self.start <= other.start < self.end:
            return True
        elif other.start <= self.start < other.end:
            return True
        else:
            return False

    def contains(self, other: "TimeRange") -> bool:
        """
        Checks whether other is a sub interval of this time range.
        """
        return self.start <= other.start and other.end <= self.end

    def union(self, other: "TimeRange"):
        if not self.intersects(other):
            raise ValueError("Cannot union non-intersecting time ranges")
        else:
            return TimeRange(
                start=min(self.start, other.start), end=max(self.end, other.end)
            )

    def intersection(self, other: "TimeRange") -> "TimeRange":
        if not self.intersects(other):
            raise ValueError("Cannot give intersection of non-intersecting time-ranges")
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
        string_repr = "-".join(re.split(r"\+|\-", ts.isoformat())[:-1])
        return f"'{string_repr} [{ts.tz}]'"
