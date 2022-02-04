import numpy as np
import os
import unittest
from main import lagrange_interpolation, hermite_cubic_interpolation, natural_cubic_interpolation, \
    natural_cubic_interpolation, periodic_cubic_interpolation
from lib import plot_function, plot_function_interpolations, plot_spline, animate, linear_animation, cubic_animation, \
    runge_function, pad_coefficients


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isfile("data.npz"):
            cls.data = np.load("data.npz", allow_pickle=True)
        else:
            raise IOError("Could not load data file 'data.npz' for tests.")

    @classmethod
    def tearDownClass(cls):
        cls.data.close()

    def test_1_lagrange_interpolation(self):
        x, y = runge_function(n=10)
        poly, base = lagrange_interpolation(x, y)
        x1, y1 = runge_function(n=4)
        poly1, base1 = lagrange_interpolation(x1, y1)

        self.assertTrue(np.allclose(pad_coefficients(poly, 10), Tests.data["t1_coeffs"]))
        self.assertTrue(np.allclose(pad_coefficients(poly1, 4), Tests.data["t1_coeffs1"]))
        for i, b in enumerate(base):
            self.assertTrue(np.allclose(pad_coefficients(b, 10), Tests.data["t1_base"][i]))
        for i, b in enumerate(base1):
            self.assertTrue(np.allclose(pad_coefficients(b, 4), Tests.data["t1_base1"][i]))

        x_runge, y_runge = runge_function(n=100)  # Runge function evaluated for "continuous" plot
        # Plot of the Runge function
        # plot_function(x_runge, y_runge)

        supports = []
        interpolations = []
        bases = []
        for i in range(3, 12):
            x_s, y_s = runge_function(n=i)  # Generate sampling points
            p, b = lagrange_interpolation(x_s, y_s)  # Generate Lagrange interpolation polynomial
            y_i = p(x_runge)  # Evaluated polynomial for "continuous" plot
            supports.append([x_s, y_s])
            interpolations.append([x_runge, y_i])
            bases.append(b)
            # Compare interpolations
            self.assertTrue(np.allclose(interpolations[i - 3][0], Tests.data["t1_interpolations"][i - 3][0]))
            self.assertTrue(np.allclose(interpolations[i - 3][1], Tests.data["t1_interpolations"][i - 3][1]))

        plot_function_interpolations([x_runge, y_runge], supports, interpolations, bases)

        # base = list(map(lambda pol: pad_coefficients(pol, 10), base))
        # base1 = list(map(lambda pol: pad_coefficients(pol, 4), base1))
        # coeffs = pad_coefficients(poly, 10)
        # coeffs1 = pad_coefficients(poly1, 4)
        # np.savez("data1", t1_interpolations=interpolations, t1_base=base, t1_base1=base1, t1_coeffs=coeffs, t1_coeffs1=coeffs1)


    def test_2_hermite_cubic_interpolation(self):
        x, y = runge_function(8)
        yp = -2.0 * x / ((1.0 + x ** 2) ** 2)
        spline = hermite_cubic_interpolation(x, y, yp)
        self.assertTrue(len(spline) == 7)
        for i, pol in enumerate(spline):
            coeffs = pad_coefficients(pol, 4)
            self.assertTrue(np.allclose(coeffs, Tests.data["t2_spline"][i]))
        plot_spline([x, y], spline)

        # spline = list(map(lambda pol: pad_coefficients(pol, 4), spline))
        # np.savez("data2", t2_spline=spline)

    def test_3_natural_cubic_animation(self):
        # x-values to be interpolated
        keytimes = np.linspace(0, 200, 11)
        # y-values to be interpolated
        keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
                     np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
        keyframes.append(keyframes[0])
        splines = []
        for i in range(11):  # Iterate over all animated parts
            x = keytimes
            y = np.array([keyframes[k][i] for k in range(11)])
            spline = natural_cubic_interpolation(x, y)
            if len(spline) == 0:
                animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
                self.fail("Natural cubic interpolation not implemented.")
            splines.append(spline)

        animate(keytimes, keyframes, cubic_animation(keytimes, splines))

    def test_4_periodic_cubic_animation(self):
        # x-values to be interpolated
        keytimes = np.linspace(0, 200, 11)
        # y-values to be interpolated
        keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
                     np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
        keyframes.append(keyframes[0])
        splines = []
        for i in range(11):  # Iterate over all animated parts
            x = keytimes
            y = np.array([keyframes[k][i] for k in range(11)])
            spline = periodic_cubic_interpolation(x, y)
            if len(spline) == 0:
                self.fail("Periodic cubic interpolation not implemented.")
            splines.append(spline)

        animate(keytimes, keyframes, cubic_animation(keytimes, splines))


if __name__ == '__main__':
    unittest.main()

