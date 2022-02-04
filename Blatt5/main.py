import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # initialize matrix with proper size
    F = np.zeros((n, n), dtype='complex128')

    # create principal term for DFT matrix
    omega = np.exp(-2 * np.pi * 1j / n)

    # fill matrix with values
    for row in range(n):
        for col in range(n):
            F[row][col] = omega ** (row * col)

    # normalize dft matrix
    F = F / np.sqrt(n)

    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """

    return np.allclose(np.eye(len(matrix)), matrix.dot(matrix.T.conj()))


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # create signals and extract harmonics out of DFT matrix
    dft = dft_matrix(n)
    for i in range(n):
        signal = np.zeros(n)
        signal[i] = 1
        sigs.append(signal)
        fsigs.append(dft.dot(signal))

    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

    # implement shuffling by reversing index bits
    shuffle = np.zeros(data.size, dtype='complex128')

    for i in range(data.size):
        temp = int(bin(i)[2:].zfill(int(np.log2(data.size)))[::-1], 2)
        shuffle[temp] = data[i]

    data = shuffle
    return data


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size

    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # first step of FFT: shuffle data
    fdata = shuffle_bit_reversed_order(fdata)

    # second step, recursively merge transforms
    foo = 1
    while n > foo:
        for i in range(foo):
            w = complex(np.exp((2 * -1j * np.pi * i) / (foo * 2)))
            for j in range(i, n, 2 * foo):
                k = w * fdata[j + foo]
                fdata[j + foo] = fdata[j] - k
                fdata[j] = fdata[j] + k
        foo = foo * 2

    # normalize fft signal
    fdata /= np.sqrt(n)

    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 2.0 * np.pi

    data = np.zeros(num_samples)

    # Generate sine wave with proper frequency
    x = x_max - x_min
    x /= num_samples - 1
    for i in range(num_samples):
        data[i] = np.sin(x * i * f)

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """
    
    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit*adata.size/sampling_rate)

    # compute Fourier transform of input data
    adata = fft(adata)

    # set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.
    adata[bandlimit_index+1:adata.size-bandlimit_index] = 0

    # compute inverse transform and extract real component
    temp = np.conjugate(fft(np.conjugate(adata)))
    adata_filtered = np.zeros(adata.size, dtype="float64")

    for i in range(temp.shape[0]):
        adata_filtered[i] = temp[i].real

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
