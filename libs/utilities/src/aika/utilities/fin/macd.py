"""
This is a simplified version of the open source code appearing here
https://github.com/gityoav/pyg-timeseries/blob/main/src/pyg_timeseries/_ewmxo.py
"""
from aika.utilities.pandas_utils import Tensor


def alpha_from_span(span):
    return 2 / (1 + span)


def ou_factor(fast_span, slow_span):
    """
    OU factor for momentum predictions.
    Calculates the variance of an OU process defined as ewma(dB, fast) - ewma(dB, slow)
    if dB is a standard Brownian Motion

    Suppose
    f = 2/(1+fast_span); F = 1-f; F2 = F^2
    s = 2/(1+slow_span); S = 1-s; S2 = S^2
    (f, s, are the alpha parameters for an ewm)
    If returns are IID and WLOG ts(0) = 0 we have that (once we flip returns)

    fast_ewma(0) = ts(0) + F * rtn(0) + F^2 * rtn(-1) + F^3 * rtn(-2) + ...
    slow_ewma(0) = ts(0) + S * rtn(0) + S^2 * rtn(-1) + S^3 * rtn(-2) + ...
    crossover(0) = (F-S) rtn(0) + (F^2-S^2) * rtn(-1)...

    The process has zero mean and variance, so assuming iid returns all the cross terms vanish:

    E(crossover^2) = \sum_{i>=1} (F^i - S^i)^2
                   = \sum_{i>=1} (F^2i + S^2i - 2 F^i * S^i)
                   = F^2 / (1 - F^2) + S^2 / (1-S^2) - 2 F*S / (1-F*S)

        (and unit variance for the returns, so the returns variance drops out).

    Parameters
    ----------
    fast : int/frac
        number of days. can also be 1/(1+days) if presented as a fraction
    slow : int/frc
        number of days. can also be 1/(1+days) if presented as a fraction
    Returns
    -------
    float
        The variance of an OU process defined as ewma(dB, fast) - ewma(dB, slow) if dB is a standard Brownian Motion
    """
    f = alpha_from_span(fast_span)
    F = 1 - f
    F2 = F**2
    s = alpha_from_span(slow_span)
    S = 1 - s
    S2 = S**2
    return (F2 / (1 - F2) + S2 / (1 - S2) - 2 * F * S / (1 - F * S)) ** 0.5


def ewma(data, span):
    return data.ewm(span=span, min_periods=3 * span).mean()


def ewm_volatility(returns: Tensor, span: int, centered=False) -> Tensor:
    if centered:
        return returns.ewm(span=span, min_periods=3 * span).std()
    else:
        return returns.pow(2).ewm(span=span, min_periods=3 * span).mean().pow(0.5)


def macd(prices: Tensor, fast_span: int, slow_span: int, vol_span: int):
    """
    res = (ewma(rtn, fast) - ewma(rtn, slow)) / (ewm_volatility(rtn, vol) * ou_factor(fast, slow))

    Parameters
    ----------
    prices: Tensor
    fast_span : int
    slow_span : int
    vol_span : int

    Returns
    -------
    An MACD signal that should be mean 0 vol 1 if the input data is brownian motion.
    """
    return (ewma(prices, fast_span) - ewma(prices, slow_span)) / (
        ewm_volatility(
            prices.ffill().diff().where(prices.notnull()), span=vol_span, centered=False
        )
        * ou_factor(fast_span, slow_span)
    )
