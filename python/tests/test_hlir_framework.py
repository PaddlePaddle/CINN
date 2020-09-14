from cinn.framework import *
import unittest
import numpy as np


class TensorTest(unittest.TestCase):
    def test_basic(self):
        tensor = Tensor()
        data = np.random.random([10, 5])
        tensor.from_numpy(data)

        self.assertTrue(np.allclose(tensor.numpy(), data))


if __name__ == "__main__":
    unittest.main()
