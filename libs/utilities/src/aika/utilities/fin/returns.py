from typing import Optional, TypeVar

import pandas as pd

from aika.utilities.pandas_utils import Tensor


class ReturnType:
    """
    When moving from prices, consider the case where there is a missing price due to a holiday. Suppose you have a
    price for monday and then a missing price on tuesday and then a price on monday. When eg calculating signal vol
    for a signal imput, we must use returns that forward fill the prices. We will not know the monday-wednesday return
    until we get a new price on wednesday. So prices (1, nan, 2) would become returns (nan, nan, 1). This is the
    causally correct moment when thinking about signal generation.

    On the other hand, when evaluating the returns to trades, only trades done on monday would experience this return,
    if signals re-evalutate on the holiday there is a risk of mistakenly experiencing this return on wednesday when
    a position could not be altered, so when thinking about "responses" or "experienced returns" we must instead
    backfill the prices. Then prices (1, nan, 2) would yield returns (nan, 1, nan). When thinking about a trading
    system, the returns experience by the signals, or the "responses" must assume returns happen as soon as possible
    to avoid any changes to the signal when the signal is non-tradeable.
    """

    CAUSAL = "causal"
    TRADING = "trading"

    ALL = {CAUSAL, TRADING}


class BarReturnIndexNames:
    START = "start"
    END = "end"


def arithmetic_bar_returns(
    prices: Tensor,
    step: int = 1,
    return_type: str = "causal",
    fill_limit: Optional[int] = None,
) -> Tensor:
    """
    This converts a set of prices into a set of bar returns, that is, returns where each period represents a bar
    from a start index to an end index. The bars will be contiguous if step is one.

    Parameters
    ----------
    prices: Tensor
        Some kind of time series that represents contract values.
    step:
        The "step size" for the diff. So on daily data a step of 7 will give you weekly returns recalculated
        every day.
    return_type:
        An entry from the class ReturnType. See documentation therein
    fill_limit:
        The maximum number of rows to back of forward fill a price.

    Returns
    -------
    Tensor: The arithmetic returns i.e. returns calculated by subtraction.
    """
    if not return_type in ReturnType.ALL:
        raise ValueError(f"return_type must be in {ReturnType.ALL}")
    elif return_type == ReturnType.CAUSAL:
        returns = (
            prices.ffill(limit=fill_limit)
            .diff(step)
            .where(prices.notnull())
            .iloc[step:]
        )
    else:
        returns = (
            prices.bfill(limit=fill_limit)
            .diff(step)
            .where(prices.notnull())
            .iloc[step:]
        )

    returns.index = pd.MultiIndex.from_arrays(
        [prices.index[:-step], prices.index[step:]],
        names=(BarReturnIndexNames.START, BarReturnIndexNames.END),
    )
    return returns
