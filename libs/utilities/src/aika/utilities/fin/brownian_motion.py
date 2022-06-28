import numpy as np
import pandas as pd


def brownian_returns(index, columns, seed=None):
    return pd.DataFrame(
        index=index,
        columns=columns,
        data=np.random.RandomState(seed=seed).randn(len(index), len(columns)),
    )


def brownian_prices(
    index,
    columns,
    vol,
    seed=None,
):
    returns = brownian_returns(index, columns, seed=seed)
    prices = returns.mul(vol).cumsum()
    return prices.sub(prices.min() - 1.0)
