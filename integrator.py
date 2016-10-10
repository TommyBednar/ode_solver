from __future__ import division
import numpy as np
import scipy.optimize as opt


def _init(x0, t0, tf, h):
    n = int((tf-t0)/h)
    t = np.linspace(t0, tf, n+1)
    x = np.zeros(t.size)
    x[0] = x0
    return n, t, x

def integrate(method, derivatives, params):
    f, dfdt, fx = derivatives
    if method == 'euler':
        return euler(f, *params)
    elif method == 'ts2':
        return ts2(f, dfdt, *params)
    elif method == 'be':
        return backwards_euler(f, fx, *params)
    elif method == 'cn':
        return trapezoidal(f, fx, *params)
    elif method == 'bdf2':
        return bdf2(f, fx, *params)
    elif method == 'ab3':
        return ab3 (f, fx, *params)
    else:
        raise ValueError('Unrecognized integration method: ' + method)

# x[n+1] - x[n] = h*f[n]
def euler(f, x0, t0, tf, h):
    n, t, x = _init(x0, t0, tf, h)
    for i in range(n):
        x[i+1] = x[i] + h * f(t[i], x[i])
    return t, x

# x[n+1]- x[n]= h*f[n] + 1/2*h^2*f'[n]
def ts2(f, dfdt, x0, t0, tf, h):
    n, t, x = _init(x0, t0, tf, h)
    for i in range(n):
        x[i+1] = x[i] + h*f(t[i], x[i]) + 1/2*h*h*dfdt(t[i], x[i])
    return t, x

# x[n+1] - x[n] = h*f[n+1]
def backwards_euler(f, fx, x0, t0, tf, h):
    n, t, x = _init(x0, t0, tf, h)
    for i in range(n):
        g = lambda u: u - x[i] - h*f(t[i+1], u)
        dg = lambda u: 1 - h*fx(t[i+1], u)
        x[i+1] = opt.newton(g, x[i], dg, tol=1e-6)
    return t, x


# x[n+1] - x[n] = 1/2*h*(f[n+1]+f[n])
def trapezoidal(f, fx, x0, t0, tf, h):
    n, t, x = _init(x0, t0, tf, h)
    for i in range(n):
        g = lambda u: u - x[i] - 1/2*h*(f(t[i+1], u) + f(t[i], x[i]))
        dg = lambda u: 1 - 1/2*h*fx(t[i+1], u)
        x[i+1] = opt.newton(g, x[i], dg, tol=1e-6)
    return t, x

# x[n+2] - 4/3*x[n+1] + 1/3*x[n] = 2/3*h*f[n+2]
def bdf2(f, fx, x0, t0, tf, h):
    n, t, x = _init(x0, t0, tf, h)
    x[1] = trapezoidal(f, fx, x0, t0, t0 + h, h)[1][1]
    for i in range(n-1):
        g = lambda u: u - 4/3*x[i+1] + 1/3*x[i] - 2/3*h*f(t[i+2], u)
        dg = lambda u: 1 - 2/3*h*fx(t[i+2], u)
        x[i+2] = opt.newton(g, x[i+1], dg, tol=1e-6)
    return t, x

# x[n+3] - x[n+2] = 1/12*h*(23*f[n+2] - 16*f[n+1] + 5*f[n])

def ab3(f, fx, x0, t0, tf, h):
    n, t, x = _init(x0, t0, tf, h)
    x[1:3] = trapezoidal(f, fx, x0, t0, t0 + 2*h, h)[1][1:3]
    for i in range(n-2):
        x[i+3] = x[i+2] + h*f(t[i+2],x[i+2])#1/12*h*(23*f(t[i+2], x[i+2]) - 16*f(t[i+1], x[i+1]) + 5*f(t[i], x[i]))
    return t, x

