import numpy as np
import pandas as pd
import pytest as pytest
from scipy.stats import chi2

from aika.utilities.fin.brownian_motion import brownian_prices
from aika.utilities.fin.macd import macd


@pytest.mark.parametrize(
    "prices, fast_span, slow_span, vol_span",
    [
        (brownian_prices(pd.RangeIndex(10000), range(10), 1.0, seed=42), 10, 20, 40),
        (brownian_prices(pd.RangeIndex(10000), range(10), 1.0, seed=42), 20, 40, 40),
    ],
)
def test_convergence(prices, fast_span, slow_span, vol_span):
    """
    There are roughly n/fast_span independent gaussian draws in a ewm macd, so the expected vol of the mean is
    sqrt(fast_span/n) (macd should normalise the vol). So assert it should be all within 4 std deviations.

    By cochrane's theorem the variance of the variance is 2 sigma ** 4 / (n-1). Since our variance should
    always be one, it follows that 4 sigma is 8 / (n-1).

    """

    result = macd(prices, fast_span, slow_span, vol_span)

    k = prices.shape[0] / fast_span  # effective degrees of freedom.
    assert result.mean().abs().lt(3 * np.sqrt(1 / k)).all()
    assert result.std().pow(2).sub(1.0).abs().lt(3 * np.sqrt(2 / (k - 1))).all()
