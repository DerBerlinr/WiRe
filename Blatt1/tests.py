import numpy as np
import unittest
from main import rotation_matrix, matrix_multiplication, compare_multiplication, inverse_rotation, machine_epsilon


class Tests(unittest.TestCase):

    def test_matrix_multiplication(self):
        a = np.random.randn(2, 2)
        c = np.random.randn(3, 3)
        self.assertTrue(np.allclose(np.dot(a, a), matrix_multiplication(a, a)))
        self.assertRaises(ValueError, matrix_multiplication, a, c)

    def test_compare_multiplication(self):
        r_dict = compare_multiplication(200, 40)
        for r in zip(r_dict["results_numpy"], r_dict["results_mat_mult"]):
            self.assertTrue(np.allclose(r[0], r[1]))

    def test_machine_epsilon(self):
        print("EPS", machine_epsilon(np.dtype(float)))
        machine_epsilon(np.dtype(np.float32))
        machine_epsilon(np.dtype(np.float64))


    def test_rotation_matrix(self):
        non_rot = np.array([[1, 2], [3, 4]])
        rot_mat = rotation_matrix(90)
        print(np.rot90(non_rot))
        print(non_rot * rot_mat)
        self.assertTrue(np.allclose(np.rot90(non_rot), np.dot(non_rot, rot_mat)))

    def test_inverse_rotation(self):
        return


if __name__ == '__main__':
    unittest.main()
