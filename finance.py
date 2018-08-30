# -*- coding: utf-8 -*-
import numpy
import math
import scipy.stats


def trapezoidal_rule(function, a, b, n):
    """This function approximates a definite integral by estimating the area under the graph
     of the integrand as a trapezoid. The arguments are as follows: function is the integrand, a and b are the
     integration bounds, and n is an integer number of strips to use. Its return value answer is an approximation of the
     definite integral of function."""
    answer = 0
    # calculate the strip size
    h = float(b - a) / n
    assert type(n) == int
    # evaluate the endpoints
    answer += function(a) + function(b)
    # evaluate the midpoints
    answer += sum([2 * function(a + k * h) for k in xrange(1, n)])
    answer *= (h / 2)
    return answer


def simpsons_rule(function, a, b, n):
    """This function is an implementation of the composite Simpson's rule. The arguments are as follows: function is
    the integrand, a and b are the integration bounds, and n is the integer number of strips to use in the
    approximation. Its return value answer is an approximation of the definite integral of function."""
    answer = 0
    # calculate the strip size
    h = float(b - a) / (2 * n)
    assert type(n) == int
    answer += function(a) + function(b)
    # evaluate the odd terms
    answer += sum([2 * function(a + (2 * k) * h) for k in xrange(1, n)])
    # evaluate the even terms
    answer += sum([4 * function(a + (2 * k - 1) * h) for k in xrange(1, n + 1)])
    answer *= h / 3
    return answer


def crude_monte_carlo(function, a, b, n):
    """This function provides a numerical approximation to an integral using pseudo-random numbers. The arguments are
    as follows: function is the integrand, a and b are the bounds of the integral and n is the integer number of
    samples. Its return value answer is an approximation of the definite integral of function."""
    assert type(n) == int
    total = sum([function(scipy.random.uniform(a, b)) for k in xrange(n)])
    answer = (total / n) * (b - a)
    return answer


def black_scholes_european_call_price(spot, strike, r, sigma, maturity):
    """This function computes the Black-Scholes option price for the fair value of a european vanilla call option.
    The arguments are as follows: spot is the spot price of the underlying asset, strike is the strike
    price, r is the risk free interest rate, sigma is the volatility of the underlying asset, and maturity is the time
    to maturity. Its return value answer is the exact fair value of the call option."""
    # No arbitrage condition
    if spot < 0:
        return None
    else:
        d_1 = ((math.log(spot / strike) + (r + sigma ** 2 / 2) * maturity)
               / (sigma * math.sqrt(maturity)))
        d_2 = d_1 - sigma * math.sqrt(maturity)
        answer = (spot * scipy.stats.norm.cdf(d_1) - strike * math.exp(-r * maturity)
                  * scipy.stats.norm.cdf(d_2))
        return answer


def black_scholes_european_put_price(spot, strike, r, sigma, maturity):
    """This function computes the put price of a european vanilla put option. The arguments are as follows: spot is
    the spot price of the underlying asset, strike is the strike price, r is the risk free interest rate, sigma is the
    volatility of the underlying asset, and maturity is the time to maturity. Its return value answer is the exact
    fair value of the put price."""
    # No arbitrage condition
    if spot < 0:
        return None
    else:
        d_1 = ((math.log(spot / strike) + (r + sigma ** 2 / 2) * maturity)
               / (sigma * math.sqrt(maturity)))
        d_2 = d_1 - sigma * math.sqrt(maturity)
        answer = (strike * math.exp(-r * maturity) * scipy.stats.norm.cdf(-d_2) - spot
                  * scipy.stats.norm.cdf(-d_1))
        return answer


def monte_carlo_european_call_price(spot, strike, r, sigma, maturity, n):
    """This function approximates the european vanilla call option price using a monte carlo approximation.
    The arguments are as follows: spot is the spot price of the underlying asset, strike is the strike price, r is the
    risk free interest rate, sigma is the volatility of the underlying asset, maturity is the time to maturity, and n
    is the number of samples to take. Its return value answer is an approximation to the fair value price of the
    call option."""
    # No arbitrage condition
    if spot < 0:
        return None
    else:
        s_factor = spot * math.exp(maturity * (r - 0.05 * sigma ** 2))
        s_current = [s_factor * math.exp(math.sqrt(sigma ** 2 * maturity)
                                         * numpy.random.normal(0, 1)) for k in xrange(n)]
        total = [max(Asset - strike, 0.0) for Asset in s_current]
        answer = scipy.mean(total) * math.exp(-r * maturity)
        return answer


def monte_carlo_european_put_price(spot, strike, r, sigma, maturity, n):
    """This function approximates the european vanilla put option price using a monte carlo approximation.
    The arguments are as follows: spot is the spot price of the underlying asset, strike is the strike price, r is the
    risk free interest rate, sigma is the volatility of the underlying asset, maturity is the time to maturity, and n
    is the number of samples to take. Its return value answer is an approximation to the fair value put price."""
    # No arbitrage condition
    if spot < 0:
        return None
    else:
        s_factor = spot * math.exp(maturity * (r - 0.05 * sigma ** 2))
        s_current = [s_factor * math.exp(math.sqrt(sigma ** 2 * maturity)
                                         * numpy.random.normal(0, 1)) for k in xrange(n)]
        total = [max(strike - Asset, 0.0) for Asset in s_current]
        answer = scipy.mean(total) * math.exp(-r * maturity)
        return answer


def black_scholes_european_call_3d(sigma, r, strike, maturity, asset_steps):
    """This function uses a finite difference method to compute solutions to the Black-Scholes partial
    differential equation. The arguments are as follows: sigma is the volatility of the underlying asset, r is the risk
    free interest rate, maturity is the time to maturity, and asset_steps controls the precision of the
    approximation. It returns an array v containing spot price, time step data, and the corresponding option
    value data."""
    ds = 2 * strike / asset_steps
    dt = 0.9 / (sigma ** 2 * asset_steps ** 2)
    time_steps = int(maturity / dt) + 1
    dt = maturity / time_steps
    v = numpy.zeros((asset_steps, time_steps))
    s = numpy.zeros(asset_steps)

    for i in range(asset_steps):
        s[i] = i * ds
        v[i, 0] = max(s[i] - strike, 0)

    for k in range(1, time_steps):
        for i in range(1, len(s) - 1):
            delta = (v[i + 1, k - 1] - v[i - 1, k - 1]) / (2 * ds)
            gamma = (v[i + 1, k - 1] - 2 * v[i, k - 1] + v[i - 1, k - 1]) / ds ** 2
            theta = -0.5 * (sigma ** 2) * s[i] ** 2 * gamma - r * s[i] * delta + r * v[i, k - 1]
            v[i, k] = v[i, k - 1] - dt * theta

        v[0, k] = v[0, k - 1] * (1 - r * dt)
        v[len(s) - 1, k] = 2 * v[len(s) - 2, k] - v[len(s) - 3, k]

    return v


def monte_carlo_asset_price_path(spot, mu, sigma, time_horizon, asset_paths,
                                 time_steps):
    """This function simulates the price of an asset undergoing a geometric brownian motion using a vectorised euler
    discretisation. The arguments are as follows: spot is the spot price of the asset, mu is the drift, sigma is the
    volatility, T is the time horizon, asset_paths is the number of asset asset paths, and time_steps is the number of
    time steps. It returns an array s containing time step data, and asset price data."""
    s = numpy.zeros((asset_paths, time_steps + 1))
    dt = time_horizon / time_steps
    s[:, 0] = spot
    epsilon = numpy.random.normal(0, 1, (asset_paths, time_steps))
    s[:, 1:] = numpy.exp((mu - 0.5 * sigma ** 2) * dt + epsilon * sigma * numpy.sqrt(dt))
    s = numpy.cumprod(s, axis=1)
    return s
