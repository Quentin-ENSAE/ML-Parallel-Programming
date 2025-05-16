import unittest
import numpy as np
import mmat

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilizing to prevent overflow
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class TestMmat(unittest.TestCase):
    def test_mmat_v0(self):
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                print("Start")
                m1 = np.random.rand(128, 128).astype(dtype)
                m2 = np.random.rand(128, 128).astype(dtype)
                m3 = np.random.rand(128, 128).astype(dtype)
                res = mmat.mmat(m1, m2, m3)
                print(m2.shape)
                print(m2.shape[1])
                #print(res[0])
                #print("-----------------------------")
                #print((m1 @ np.transpose(m2))[0])
                #np.testing.assert_allclose((m1 @ np.transpose(m2)) / (m2.shape[1] ** 0.5), res, atol=1e-4)
                np.testing.assert_allclose(softmax((m1 @ np.transpose(m2)) / m2.shape[1] ** 0.5) @ m3, res, atol=1e-4)

    def test_mmat_v1(self):
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                m1 = np.random.rand(128, 128).astype(dtype)
                m2 = np.random.rand(128, 128).astype(dtype)
                m3 = np.random.rand(128, 128).astype(dtype)
                res = mmat.mmat(m1, m2, m3, version=1)
                np.testing.assert_allclose(m1 @ m2, res, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
