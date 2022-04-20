from typing import Union

import pandas as pd


# a fake class that returns a pandas timestamp guaranteed to have correct timezone info
def Timestamp(
    ts_input: Union[str, pd.Timestamp],
) -> pd.Timestamp:

    if isinstance(ts_input, pd.Timestamp):
        if ts_input.tz is None:
            return pd.Timestamp(ts_input=ts_input, tz="UTC")
        else:
            return ts_input
    elif isinstance(ts_input, str):
        split = ts_input.split("[")
        if len(split) == 2:
            return pd.Timestamp(ts_input=split[0], tz=split[1][:-1])
        else:
            return pd.Timestamp(ts_input=ts_input, tz="UTC")
    else:
        raise ValueError("Input not recognised")
