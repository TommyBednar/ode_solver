from __future__ import division
import numpy as np


def _init(x0, t0, tf, h):
    n = int((tf-t0)/h)
    t = np.linspace(t0, tf, n+1)
    x = np.zeros(t.size)
    x[0] = x0
    return n, t, x


def integrate(method, derivatives, params):
    f, df = derivatives
    if method == 'euler':
        return euler(f, *params)
    elif method == 'ts2':
        return ts2(f, df, *params)
    raise ValueError('Unrecognized integration method: ' + method)

# x[n+1] - x[n] = h*f[n]
def euler(f, x0, t0, tf, h):
    n, t, x = _init(x0, t0, tf, h)
    for i in range(n):
        x[i+1] = x[i] + h * f(t[i], x[i])
    return t, x

# x[n+1]- x[n]= h*f[n] + 1/2*h^2*f'[n]
def ts2(f, df, x0, t0, tf, h):
    n, t, x = _init(x0, t0, tf, h)
    for i in range(n):
        x[i+1] = x[i] + h*f(t[i], x[i]) + 0.5*h*h*df(t[i], x[i])
    return t, x

# x[n+1] - x[n] = h*f[n+1]
def backwards_euler():
    pass

# x[n+1] - x[n] = 1/2*h*(f[n+1]+f[n])
def trapezoidal():
    pass

# x[n+2] - 4/3*x[n+1] + 1/3*x[n] = 2/3*h*f[n+2]
def bdf2():
    pass

# x[n+3] - x[n+2] = 1/12*h*(23*f[n+2] - 48*f[n+1] 5*f[n])
def ab3():
    pass