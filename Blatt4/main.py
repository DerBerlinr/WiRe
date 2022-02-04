import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    # Generate Lagrange base polynomials and interpolation polynomial
    for i in range(len(x)):
        func = np.poly1d(1)
        for j in range(len(x)):
            if j == i:
                pass
            else:
                func *= np.poly1d([1, -x[j]]) / (x[i] - x[j])
        base_functions.append(func)

    for index, function in enumerate(base_functions):
        polynomial += function*y[index]

    return polynomial, base_functions


def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # compute piecewise interpolating cubic polynomials

    for i in range(len(x)):
        if i == len(x)-1:
            break
        M = np.zeros((4, 4))
        M[0, :] = x[i] ** 3, x[i] ** 2, x[i], 1
        M[1, :] = x[i+1] ** 3, x[i+1] ** 2, x[i+1], 1
        M[2, :] = 3*x[i] ** 2, 2*x[i], 1, 0
        M[3, :] = 3*x[i+1] ** 2, 2*x[i+1], 1, 0

        b = np.array([y[i], y[i+1], yp[i], yp[i+1]])

        sol = np.linalg.solve(M, b)
        spline.append(np.poly1d(sol))

    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # construct linear system with natural boundary conditions
    A = np.zeros((4*(len(y)-1), 4*(len(x)-1)))

    for i in range(len(x)-1):
        A[4*i, 4*i] = x[i] ** 3
        A[4*i, 4*i+1] = x[i] ** 2
        A[4*i, 4*i+2] = x[i]
        A[4*i, 4*i+3] = 1

        A[4*i+1, 4*i] = x[i+1] ** 3
        A[4*i+1, 4*i+1] = x[i+1] ** 2
        A[4*i+1, 4*i+2] = x[i+1]
        A[4*i+1, 4*i+3] = 1

    for i in range(len(x)-2):
        A[4*i+2, 4*i] = (x[i+1] ** 2) * 3
        A[4*i+2, 4*i+1] = x[i+1] * 2
        A[4*i+2, 4*i+2] = 1

        A[4*i+2, 4*i+4] = (x[i+1] ** 2) * -3
        A[4*i+2, 4*i+5] = x[i+1] * -2
        A[4*i+2, 4*i+6] = -1

        A[4*i+3, 4*i] = x[i+1] * 6
        A[4*i+3, 4*i+1] = 2

        A[4*i+3, 4*i+4] = x[i+1] * -6
        A[4*i+3, 4*i+5] = -2

    A[-2, 0] = x[0] * 6
    A[-2, 1] = 2

    A[-1, -3] = 2
    A[-1, -4] = x[len(x) - 1] * 6

    b = np.zeros(((len(y)-1) * 4))
    for i in range(0, len(y)-1):
        b[i*4] = y[i]
        b[i*4+1] = y[i+1]

    # solve linear system for the coefficients of the spline
    sol = np.linalg.solve(A, b)

    # extract local interpolation coefficients from solution
    spline = []
    for i in range(len(x)-1):
        spline.append(np.poly1d(sol[i*4: i*4+4]))

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)

    # construct linear system with periodic boundary conditions
    A = np.zeros((4*(len(y)-1), 4*(len(x)-1)))

    for i in range(len(x)-1):
        A[4*i, 4*i] = x[i] ** 3
        A[4*i, 4*i+1] = x[i] ** 2
        A[4*i, 4*i+2] = x[i]
        A[4*i, 4*i+3] = 1

        A[4*i+1, 4*i] = x[i+1] ** 3
        A[4*i+1, 4*i+1] = x[i+1] ** 2
        A[4*i+1, 4*i+2] = x[i+1]
        A[4*i+1, 4*i+3] = 1

    for i in range(0, x.size - 2):
        A[4*i+2, 4*i] = (x[i+1] ** 2) * 3
        A[4*i+2, 4*i+1] = x[i+1] * 2
        A[4*i+2, 4*i+2] = 1

        A[4*i+2, 4*i+4] = (x[i+1] ** 2) * -3
        A[4*i+2, 4*i+5] = x[i+1] * -2
        A[4*i+2, 4*i+6] = -1

        A[4*i+3, 4*i] = x[i+1] * 6
        A[4*i+3, 4*i+1] = 2

        A[4*i+3, 4*i+4] = x[i+1] * -6
        A[4*i+3, 4*i+5] = -2

    A[-2, 0] = (x[0] ** 2) * 3
    A[-2, 1] = x[0] * 2
    A[-2, 2] = 1

    A[-1, 0] = x[0] * 6
    A[-1, 1] = 2

    A[-2, -4] = (x[x.size-1] ** 2) * -3
    A[-2, -3] = x[x.size-1] * -2
    A[-2, -2] = -1

    A[-1, -4] = x[x.size-1] * -6
    A[-1, -3] = -2

    b = np.zeros(((y.size-1) * 4))
    for i in range(0, y.size-1):
        b[i*4] = y[i]
        b[i*4+1] = y[i+1]

    # solve linear system for the coefficients of the spline
    result = np.linalg.solve(A, b)

    # extract local interpolation coefficients from solution
    spline = []
    for i in range(0, x.size-1):
        spline.append(np.poly1d(result[i*4: i*4+4]))

    return spline


if __name__ == '__main__':

    x = np.array( [1.0, 2.0, 3.0, 4.0])
    y = np.array( [3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation( x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])]*5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
