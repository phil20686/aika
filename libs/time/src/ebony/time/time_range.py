from ebony.time.timestamp import Timestamp
import pandas as pd

from ebony.utilities.pandas_utils import Tensor

RESOLUTION = pd.Timedelta(nanoseconds=1)


class TimeRange:
    """
    Represents a half-open interval (closed at the end) between two timestamps. Intended to work with ordered pandas
    time-indexed dataframes, an are valid only across the same intervals as pd.Timestamp.
    """

    strf_format = "'%Y-%m-%dT%H:%M:{seconds} [{tz}]'"

    @classmethod
    def _timestamp_repr(cls, ts: pd.Timestamp):
        # TODO : Really this belongs on the timestamp class.
        seconds = ts.second + ts.microsecond / 1e6 + ts.nanosecond / 1e9
        if seconds < 10.0:
            seconds_repr = f"0{seconds}"
        else:
            seconds_repr = str(seconds)
        # this works because we are guaranteed olsen timezones.
        return ts.strftime(cls.strf_format).format(seconds=seconds_repr, tz=ts.tz)

    def __init__(self, start: pd.Timestamp, end: pd.Timestamp):
        if start is None:
            self.start = pd.Timestamp.min.tz_localize("UTC")
        else:
            self.start = Timestamp(start)
        if end is None:
            self.end = pd.Timestamp.max.tz_localize("UTC")
        else:
            self.end = Timestamp(end)
        if not self.start <= self.end:
            raise ValueError("The start time must be before the end time")

    def __eq__(self, other):
        if isinstance(other, TimeRange):
            return self.start == other.start and self.end == other.end
        else:
            return False

    def __hash__(self):
        return hash((self.start, self.end))

    def view(self, tensor: Tensor, level=None) -> Tensor:
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
        if level is None:
            range_slice = slice(self.start + RESOLUTION, self.end)
            if isinstance(tensor, pd.Series):
                return tensor.loc[range_slice]
            elif isinstance(tensor, pd.DataFrame):
                return tensor.loc[range_slice, :]
        else:
            # /TODO I couldnt get it to make an index slice of an arbitary number
            # of levels. in a clean way, this is good enough for now.
            return tensor.loc[
                tensor.index.get_level_values(level).map(lambda x: x in self)
            ]

    def __contains__(self, item):
        return self.start < item <= self.end

    def __repr__(self):
        """
        For ease of use in notebooks it should repr into something that can then execute, i.e.
        it maintains the contract eval(repr(tr)) == tr
        """
        return f"TimeRange({self._timestamp_repr(self.start)}, {self._timestamp_repr(self.end)})"
