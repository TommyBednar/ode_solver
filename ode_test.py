from __future__ import division
from functools import partial
import integrator
import math
import matplotlib.pyplot as plt
import numpy as np


def f(k, t, x):
    return k*x

def df(k, t, x):
    return k*k*x

def g(t, x):
    return math.cos(t)*x

def dg(t, x):
    return (math.cos(t)**2 - math.sin(t))*x

def run_test_problem(test_func, description, integration_params):
    x_exact, f_test, df_test = test_func
    method, test_name = description
    hs, x0, t0, tf = integration_params
    plt.figure()
    plt.title("{} on {}".format(*description))
    for h in hs:
        derivatives = f_test, df_test
        params = x0, t0, tf, h
        t, x = integrator.integrate(method, derivatives, params)
        plt.plot(t, x, label='h = {}'.format(h))
    plt.plot(t, x_exact(t), label='exact')
    plt.legend()


def main():
    x0 = 1
    t0, tf = 0, 5

    for method in 'euler', 'ts2':
        ks = [-1, -10, -50]
        hs = [0.5, 0.1, 0.01]
        for k in ks:
            test_func = lambda t: np.exp(k*t), partial(f,k), partial(df,k)
            description = method, 'exp({}t)'.format(k)
            integration_params = hs, x0, t0, tf
            run_test_problem(test_func, description, integration_params)

        hs = [0.2, 0.1, 0.05, 0.025]
        test_func = lambda t: np.exp(np.sin(t)), g, dg
        description = method, 'exp(sin(t))'
        integration_params = hs, x0, t0, tf
        run_test_problem(test_func, description, integration_params)

    plt.show()

if __name__ == '__main__':
    main()
