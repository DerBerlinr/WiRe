
import time
import matplotlib.pyplot as plt
import numpy as np



def timedcall(fn, *args):
    """
    Run a function and measure execution time.

    Arguments:
    fn : function to be executed
    args : arguments to function fn

    Return:
    dt : execution time
    result : result of function

    Usage example:
      You want to time the function call "C = foo(A,B)".
       --> "T, C = timedcall(foo, A, B)"
    """

    t0 = time.time()
    result = fn(*args)
    t1 = time.time()

    dt = t1 - t0
    return dt, result



def plot_2d(x_data, y_data, labels, title, x_axis, y_axis, x_range):

    plt.figure()
    for i, label in enumerate(labels):
        plt.loglog(x_data, y_data[i], label=label)
    plt.grid()

    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel(x_axis)
    plt.xlim(x_range[0], x_range[1])
    plt.ylabel(y_axis)
    plt.show()

