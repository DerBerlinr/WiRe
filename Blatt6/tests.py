import os
import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib import animation

from lib import fpoly, dfpoly, fractal_functions, generate_sampling, get_colors, generate_cylinder, load_object, \
    prepare_visualization, update_visualization, calculate_abs_gradient
from main import find_root_bisection, find_root_newton, generate_newton_fractal, surface_area, surface_area_gradient, gradient_descent_step


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # pass
        if os.path.isfile("data/data.npz"):
            cls.data = np.load("data/data.npz")
        else:
            raise IOError("Could not load data file 'data.npz' for tests.")

    @classmethod
    def tearDownClass(cls):
        # pass
        cls.data.close()

    def test_1_find_root_bisection(self):
        x0 = find_root_bisection(lambda x: x ** 2 - 2, np.float64(-1.0), np.float64(2.0))
        self.assertTrue(np.isclose(x0, np.sqrt(2)))
        x1 = find_root_bisection(fpoly, np.float64(-1.0), np.float64(5.0))
        x2 = find_root_bisection(fpoly, np.float64(1.0), np.float64(4.0))
        x3 = find_root_bisection(fpoly, np.float64(4.0), np.float64(5.0))
        x = np.linspace(-1.0, 5.5, 1000)
        plt.plot(x, fpoly(x))
        plt.plot([x1, x2, x3], [0.0] * 3, 'ro')
        plt.grid(True)
        plt.show()

    def test_2_find_root_newton(self):
        x0, i0 = find_root_newton(lambda x: x ** 2 - 2, lambda x: 2 * x, np.float64(10.0))
        x1, i1 = find_root_newton(lambda x: x ** 2 - 2, lambda x: 2 * x, np.float64(5.0))
        x2, i2 = find_root_newton(lambda x: x ** 2 - 2, lambda x: 2 * x, np.float64(0.1))
        self.assertTrue(np.allclose(np.array([x0, x1, x2]), np.array([np.sqrt(2)] * 3)))

        x0, i0 = find_root_newton(fpoly, dfpoly, np.float64(-1.0))
        x1, i1 = find_root_newton(fpoly, dfpoly, np.float64(2.0))
        x2, i2 = find_root_newton(fpoly, dfpoly, np.float64(5.0))
        self.assertTrue(np.allclose(np.array([x0, x1, x2]), np.array([0.335125152578, 2.61080833945, 4.79087461944])))
        x = np.linspace(-1.0, 5.5, 1000)
        plt.plot(x, fpoly(x))
        plt.plot([x0, x1, x2], [0.0] * 3, 'ro')
        plt.grid(True)
        plt.show()

    def test_3_generate_newton_fractal(self):
        size = 100 # size of the image
        max_iterations = 200

        for c, el in enumerate(fractal_functions[:]):
            f, df, roots, borders, name = el
            sampling, size_x, size_y = generate_sampling(borders, size)
            res = generate_newton_fractal(f, df, roots, sampling, n_iters_max=max_iterations)
            colors = get_colors(roots)

            # Generate image
            img = np.zeros((sampling.shape[0], sampling.shape[1], 3))
            for i in range(size_y):
                for j in range(size_x):
                    if res[i, j][1] <= max_iterations:
                        img[i, j] = colors[res[i, j][0]] / max(1.0, res[i, j][1] / 6.0)

            plt.imsave('data/fractal_' + name + '.png', img)
            self.assertTrue(np.allclose(self.data["fr_" + str(c)], img))
            # np.savez("data"+name, fr=img)

    def test_4_surface_area(self):
        nc = 32 # number of elements per layer
        nz = 12 # number of layers
        v, f, c = generate_cylinder(16, 8)
        a = surface_area(v, f)
        self.assertTrue(np.isclose(a, 4.99431224361))

    def test_5_surface_area_gradient(self):
        v, f, c = load_object("data/wave")
        gradient =  surface_area_gradient(v, f)
        gradient = gradient.flatten()
        gradient = gradient / np.linalg.norm(gradient)
        reference = self.data["gradient"].flatten()
        reference = reference / np.linalg.norm(reference)
        self.assertTrue(np.allclose(reference, gradient))
        #np.savez("datagrad", grad=gradient)

    def test_6_gradient_descent_step(self):
        model = "wave" #"cube", "moebius", "enneper"
        v, f, c = load_object("data/" + model)
        #v, f, c = generate_cylinder(32, 12)
        eps = 1e-6

        fig, surf, ax, limits = prepare_visualization(v, f)

        # Set up formatting for the movie files
        write_video = False
        if write_video:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, bitrate=2000)

        # Update step is called by animation until gradient_decent_step returns converged=True.
        # This is like running a while loop and executing gradient descent steps in the loop
        def update(i, v, f, c, surf):
            # Calculate one gradient descent step
            converged, area, vi, gradient = gradient_descent_step(v, f, c, eps)
            np.copyto(v, vi) #Update the v inplace
            # Calculate gradient length per triangle for visualization
            abs_gradient = calculate_abs_gradient(gradient, f, c)

            # Update visualization
            update_visualization(v, f, abs_gradient, limits, ax, False)

            # Debug output
            #print(area, converged)
            if converged:
                plt.close()
                self.assertTrue(abs(area - 4.573) < 1e-2)
            return surf,

        ani = animation.FuncAnimation(fig, update, fargs=(v, f, c, surf), interval=30, blit=False)
        if write_video:
            ani.save(model + '.mp4', writer=writer)
        plt.show()


#    def test_4_posterize(self):
#        lion = np.asarray(image.imread("data/lion_small.png"))
#        n_colors = 5
#        posterized = posterize(lion, n_colors, 42)

#        image.imsave("data/lion_posterized.png", posterized)
#        unique_pixels = np.vstack({tuple(r) for r in posterized.reshape(-1, 3)})
#        self.assertTrue(unique_pixels.shape[0] == n_colors)
#        self.assertTrue(np.linalg.norm(lion - posterized) < 9.0)
#        # Close to reference solution (not mandatory)
#        # self.assertTrue(np.allclose(posterized, self.data["t4_lion"]))
#        # np.savez("data2", t4_lion=posterized)


if __name__ == '__main__':
    unittest.main()

