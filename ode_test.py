from __future__ import division
from functools import partial
import integrator
import math
import matplotlib.pyplot as plt
import numpy as np


def f(k, t, x):
    return k*x

def dfdt(k, t, x):
    return k*k*x

def fx(k, t, x):
    return k

def g(t, x):
    return math.cos(t)*x

def dgdt(t, x):
    return (math.cos(t)**2 - math.sin(t))*x

def gx(t, x):
    return math.cos(t)

def run_test_problem(test_func, description, integration_params):
    x_exact, f_test, dfdt_test, fx_test = test_func
    method, test_name = description
    hs, x0, t0, tf = integration_params
    plt.figure()
    plt.title("{} on {}".format(*description))
    for h in hs:
        derivatives = f_test, dfdt_test, fx_test
        params = x0, t0, tf, h
        t, x = integrator.integrate(method, derivatives, params)
        xe = x_exact(t)
        plt.plot(x.real, x.imag, label='h = {}'.format(h))
    plt.plot(xe.real, xe.imag, label='exact')
    plt.legend()


def main():
    x0 = 1
    t0, tf = 0, 10

    for method in ['midpoint', 'ab3']:
        k = 6j
        hs = [0.2, 0.15, 0.1, 0.08]
        test_func = lambda t: np.exp(k*t), partial(f,k), partial(dfdt,k), partial(fx,k)
        description = method, 'exp({}t)'.format(k)
        integration_params = hs, x0, t0, tf
        run_test_problem(test_func, description, integration_params)

    plt.show()

if __name__ == '__main__':
    main()
