import numpy as np
import pandas as pd
import pytest as pytest

from aika.utilities.fin.brownian_motion import brownian_prices
from aika.utilities.fin.macd import macd


@pytest.mark.parametrize(
    "prices",
    [
        brownian_prices(pd.RangeIndex(10000), range(10), 0.1, seed=42),
        brownian_prices(pd.RangeIndex(10000), range(10), 1.0, seed=43),
        brownian_prices(pd.RangeIndex(10000), range(10), 10.0, seed=44),
        brownian_prices(pd.RangeIndex(10000), range(10), 100.0, seed=45),
        brownian_prices(pd.RangeIndex(1000), range(10), 0.1, seed=46),
        brownian_prices(pd.RangeIndex(1000), range(10), 1.0, seed=47),
        brownian_prices(pd.RangeIndex(1000), range(10), 10.0, seed=48),
        brownian_prices(pd.RangeIndex(1000), range(10), 100.0, seed=49),
    ],
)
@pytest.mark.parametrize(
    "fast_span, slow_span, vol_span",
    [
        (10, 20, 100),
        (20, 40, 100),
    ],
)
def test_convergence(prices, fast_span, slow_span, vol_span):
    """
    There are roughly n/fast_span independent gaussian draws in a ewm macd, so the expected vol of the mean is
    sqrt(fast_span/n) (macd should normalise the vol). So assert it should be all within 4 std deviations.

    By cochrane's theorem the variance of the variance of a normal is 2 sigma ** 4 / (n-1). Since our variance should
    always be one, it follows that 4 sigma is 4 * sqrt(2 / (n-1)).
    """

    result = macd(prices, fast_span, slow_span, vol_span)

    k = (  # correct for the nans due to the min_period for the ewm's to burn in
        prices.shape[0] - 3 * max([vol_span, fast_span, slow_span])
    ) / fast_span  # effective degrees of freedom.
    assert result.mean().abs().lt(4 * np.sqrt(1 / k)).all()
    assert result.std().pow(2).sub(1.0).abs().lt(4 * np.sqrt(2 / (k - 1))).all()
