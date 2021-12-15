import numpy
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # Test if shape of matrix and vector is compatible and raise ValueError if not
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not quadratic")
    if A.shape[1] != b.shape[0]:
        raise ValueError("shapes of A and b do not comply")

    # Perform gaussian elimination
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1):
        if A[i][i] == 0.0:
            raise ValueError("Divide by zero")
        """for l in range(i, n-1):
            if use_pivoting:
                if abs(A[i, l]) > abs(A[l, l]):
                    A[l], A[i] = A[i], A[l]"""
        for j in range(i + 1, n):
            ratio = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] = A[j][k] - ratio * A[i][k]
            b[j] = b[j] - ratio * b[i]

    # Back Substitution
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        sum = b[i]
        for j in range(i + 1, n):
            sum = sum - A[i, j] * x[j]
        x[i] = sum / A[i, i]

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # Test if shape of matrix and vector is compatible and raise ValueError if not
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not quadratic")
    if A.shape[1] != b.shape[0]:
        raise ValueError("shapes of A and b do not comply")
    # Calculate rank of matrix
    (m, n) = A.shape
    rank = 0
    for row in A:
        for element in row:
            if element != 0:
                break
            rank += 1
    full_rank = False

    if rank == m:
        full_rank = True

    if full_rank and n < m:
        raise ValueError("There are infinite solutions")
    if rank < n and rank < m:
        raise ValueError("There are either infinite or no solutions")

    # Initialize solution vector with proper size
    x = np.zeros(m)

    # Run back-substitution and fill solution vector, TODO: raise ValueError if no/infinite solutions exist
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        tmp_sum = b[i]
        for j in range(i + 1, n):
            tmp_sum = tmp_sum - A[i, j] * x[j]
        x[i] = tmp_sum / A[i, i]

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if n != m:
        raise ValueError("The matrix is not quadratic")
    for row in range(n):
        for element in range(n):
            if M[row][element] != M[element][row]:
                raise ValueError("The matrix is not symmetric")

    # build the factorization and raise a ValueError in case of a non-positive definite input matrix
    L = np.zeros((n, n))

    for i in range(n):
        for k in range(i + 1):
            tmp_sum = sum(L[i, j] * L[k, j] for j in range(k))

            if i > k:  # Diagonal elements
                L[i, k] = tmp_sum / M[k, k]
            elif tmp_sum > 0:
                L[i, i] = np.sqrt(tmp_sum)
            else:
                raise ValueError("not symmetric positive definite")

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    if n != m:
        raise ValueError("Input matrix not quadratic")
    for row in range(n-1):
        for elem in range(n-1):
            if elem > row:
                if L[row, elem] != 0:
                    raise ValueError("INPUT NOT VALID")

    # TODO Solve the system by forward- and backsubstitution
    y = np.zeros(m)

    for i in range(len(L)):
        y[i] = (y[i] - np.dot(y, L[i])) / float(L[i][i])

    L_t = np.transpose(L)

    x = back_substitution(L_t, y)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # Initialize system matrix with proper size
    L = np.zeros((n_rays*n_shots, n_grid*n_grid))
    # Initialize intensity vector
    g = np.zeros(n_rays*n_shots)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    for iteration in range(n_shots):
        theta = np.pi * (iteration/n_shots)
        # Take a measurement with the tomograph from direction r_theta.
        # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
        # ray_indices: indices of rays that intersect a cell
        # isect_indices: indices of intersected cells
        # lengths: lengths of segments in intersected cells
        # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        """print("Intensities: ", intensities)
        print("Ray-indicies: ", ray_indices)
        print("Isect-indices: ", isect_indices)
        print("Lengths: ", lengths)"""

        offset = iteration * n_rays

        for ray in range(n_rays):
            g[offset + ray] = intensities[ray]

        n_intersect = len(ray_indices)
        for intersection in range(0, n_intersect):
            cell, ray, length = isect_indices[intersection], ray_indices[intersection], lengths[intersection]
            L[offset + ray, cell] = length

    return [L, g]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [A, b] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # Cholesky decomposition
    A_T_A = A.T @ A
    A_T_b = A.T @ b
    L = np.linalg.cholesky(A_T_A)
    y = np.linalg.solve(L, A_T_b)
    x = np.linalg.solve(L.T, y)

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    i = 0
    for row in range(n_grid):
        for elem in range(n_grid):
            tim[row, elem] = x[i]
            i += 1

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
