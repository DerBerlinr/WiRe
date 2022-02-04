import os
import numpy as np
import unittest
import time
import matplotlib.pyplot as plt


from lib import idft, dft, ifft, plot_harmonics, read_audio_data, write_audio_data
from main import dft_matrix, is_unitary, create_harmonics, shuffle_bit_reversed_order, fft, \
    generate_tone, low_pass_filter


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

    def test_1_dft_matrix(self):
        dft5 = dft_matrix(4)
        print(dft5)
        dft1 = dft_matrix(n=16)
        self.assertTrue(dft1.shape[0] == dft1.shape[1] == 16)
        self.assertTrue(np.allclose(dft1, Tests.data["t1_dft1"]))

        dft2 = dft_matrix(n=64)
        self.assertTrue(dft2.shape[0] == dft2.shape[1] == 64)
        self.assertTrue(np.allclose(dft2, Tests.data["t1_dft2"]))

        signal = np.random.rand(64)
        self.assertTrue(np.allclose(idft(dft(signal)), signal))
        self.assertTrue(np.allclose(np.fft.fft(signal)/np.sqrt(signal.size), dft(signal)))

#        np.savez("data", t1_dft1=dft1, t1_dft2=dft2)

    def test_2_is_unitary(self):
        self.assertFalse(is_unitary(Tests.data["t2_m1"]))

        signal = np.random.rand(64)
        self.assertTrue(np.allclose(idft(dft(signal, True)), signal))
        self.assertTrue(is_unitary(Tests.data["t2_m2"]))

        m1 = np.random.rand(16, 16)
        m2 = dft_matrix(32)
#        np.savez("data1", t2_m1=m1, t2_m2=m2)

    def test_3_create_harmonics(self):
        s1, fs1 = create_harmonics(16)
        self.assertTrue(len(s1) == len(fs1) == 16)
        self.assertTrue(np.allclose(s1, Tests.data["t3_s1"]))
        self.assertTrue(np.allclose(fs1, Tests.data["t3_fs1"]))

        s2, fs2 = create_harmonics()
        self.assertTrue(len(s2) == len(fs2) == 128)
        self.assertTrue(np.allclose(s2, Tests.data["t3_s2"]))
        self.assertTrue(np.allclose(fs2, Tests.data["t3_fs2"]))

        plot_harmonics(s2, fs2)

#        np.savez("data3", t3_s1=s1, t3_fs1=fs1, t3_s2=s2, t3_fs2=fs2)

    def test_4_shuffle_bit_reversed_order(self):
        d1 = shuffle_bit_reversed_order(np.linspace(0, 15, 16))
        d2 = shuffle_bit_reversed_order(np.linspace(0, 8191, 8192))
        self.assertTrue(np.allclose(d1, Tests.data["t4_d1"]))
        self.assertTrue(np.allclose(d2, Tests.data["t4_d2"]))

#        np.savez("data4", t4_d1=d1, t4_d2=d2)

    def test_5_fft(self):
        data = np.random.randn(128)
        data1 = ifft(fft(data))
        self.assertTrue(np.allclose(data, data1))
        self.assertTrue(np.allclose(fft(data), np.fft.fft(data)/np.sqrt(data.size)))

        lens = [4, 16, 32, 64, 128, 256]
        n_rand = 10
        errs = np.zeros(len(lens))

        for i, clen in enumerate(lens):
            # print("Testing FFT n=%d" % clen)
            for j in range(n_rand):
                data = np.random.randn(clen)
                data1 = ifft(fft(data))
                errs[i] += np.linalg.norm(data1 - data)

        errs /= n_rand

        plt.figure()
        plt.plot(lens, errs, '-rx')
        plt.xlim(lens[0], lens[len(lens) - 1])
        plt.xlabel("Size of input")
        plt.ylabel("Error")
        plt.show()

        lens = [4, 16, 32, 64, 128, 256]
        n_rand = 10
        times = np.zeros((2, len(lens)))

        for i, clen in enumerate(lens):
            for j in range(n_rand):
                data = np.random.randn(clen)

                t0 = time.time()
                r1 = dft(data)
                t1 = time.time()
                times[0, i] += t1 - t0

                t0 = time.time()
                r2 = fft(data)
                t1 = time.time()
                times[1, i] += t1 - t0

        times /= n_rand

        plt.figure()
        plt.plot(lens, times[0, :], '-bx')
        plt.plot(lens, times[1, :], '-rx')
        plt.legend(['DFT', 'FFT'])
        plt.xlim(lens[0], lens[len(lens) - 1])
        plt.xlabel("Size of input")
        plt.ylabel("Execution time")
        plt.show()

    def test_6_generate_tone(self):
        mid_c = generate_tone()
        t42 = generate_tone(42.0)

        self.assertTrue(np.allclose(mid_c, Tests.data["t6_midc"]))
        self.assertTrue(np.allclose(t42, Tests.data["t6_42"]))
        write_audio_data('./data/mid-c.wav', mid_c, 44100)

#        np.savez("data6", t6_midc=mid_c, t6_42=t42)

    def test_7_low_pass_filter(self):
        # see http://www.ee.columbia.edu/~dpwe/sounds for more example sounds
        adata, rate = read_audio_data('./data/speech.wav')
        adata = adata[0:2**15]

        adata_filtered = low_pass_filter(adata, sampling_rate=rate)
        write_audio_data('data/speech-filtered.wav', adata_filtered, rate)


if __name__ == '__main__':
    unittest.main()

