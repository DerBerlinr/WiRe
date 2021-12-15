import datetime
import unittest

import matplotlib.pyplot as plt
import numpy as np

from main import compute_tomograph, gaussian_elimination, back_substitution, compute_cholesky, solve_cholesky


class Tests(unittest.TestCase):
    def test_gaussian_elimination(self):
        A = np.random.randn(4, 4)
        x = np.random.rand(4)
        b = np.dot(A, x)
        A_elim, b_elim = gaussian_elimination(A, b)
        self.assertTrue(np.allclose(np.linalg.solve(A_elim, b_elim), x))  # Check if system is still solvable
        self.assertTrue(np.allclose(A_elim, np.triu(A_elim)))  # Check if matrix is upper triangular

    def test_back_substitution(self):
        A = np.random.randn(4, 4)
        b = np.random.rand(4)
        # A = np.array([[11, 44, 1],
        #                 [0.1, 0.4, 3],
        #                 [0, 1, -1]])
        # b = np.array([1, 1, 1])

        A_elim, b_elim = gaussian_elimination(A, b)
        x = back_substitution(A_elim, b_elim)

        x_np = np.linalg.solve(A, b)
        self.assertTrue(np.allclose(x, x_np))

    def test_cholesky_decomposition(self):
        M = np.array([[4, 6, 2],
                      [6, 10, 5],
                      [2, 5, 21]])

        L = compute_cholesky(M)
        L_np = np.linalg.cholesky(M)

        self.assertTrue(np.allclose(L, L_np))

    def test_solve_cholesky(self):
        A = np.array([[4, 6, 2],
                      [6, 10, 5],
                      [2, 5, 21]])
        x = np.random.rand(3)
        b = np.dot(A, x)
        L = compute_cholesky(A)
        self.assertTrue(np.allclose(L, np.tril(L)))  # Check if matrix is lower triangular

        solution = solve_cholesky(L, b)
        self.assertTrue(np.allclose(solution, x))  # Check if system is still solvable
        pass

    def test_compute_tomograph(self):
        t = datetime.datetime.now()
        print("Start time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Compute tomographic image
        n_shots = 64  # 128
        n_rays = 64  # 128
        n_grid = 64  # 64
        tim = compute_tomograph(n_shots, n_rays, n_grid)

        t = datetime.datetime.now()
        print("End time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Visualize image
        plt.imshow(tim, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0],
                   origin='lower', interpolation='nearest')
        plt.gca().set_xticks([-1, 0, 1])
        plt.gca().set_yticks([-1, 0, 1])
        plt.gca().set_title('%dx%d' % (n_grid, n_grid))

        plt.show()


if __name__ == '__main__':
    unittest.main()