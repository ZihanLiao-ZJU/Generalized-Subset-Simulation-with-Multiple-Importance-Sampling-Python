import unittest
import numpy as np
from Limit_state_functions.benchmark import Benchmark
from Support_Fun.numerics import mround, mstd, quantile, source_logsum
from GSS_MIS_model import select_seeds, support
from Support_Fun.random import Random


class CoreTests(unittest.TestCase):
    def test_lsf(self):
        x = np.array([[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(
            Benchmark(1).evaluate(x), [[-0.01, 1.99], [0.31, 0.71]]
        )
        z = Benchmark(2).evaluate(x)
        np.testing.assert_allclose(
            z, [[3, 3 - 2**0.5], [7 / 2**0.5] * 2, [3, 3 + 2**0.5], [7 / 2**0.5] * 2]
        )

    def test_round_std_quantile(self):
        np.testing.assert_array_equal(mround([0.5, 1.5, 2.5]), [1, 2, 3])
        np.testing.assert_array_equal(mstd(np.ones((2, 1))), [0, 0])
        np.testing.assert_allclose(quantile(np.arange(1, 11)[None, :], 0.1), [1.5])

    def test_strict(self):
        np.testing.assert_array_equal(
            support(np.array([[0.0, -1.0, 1.0]]), 0, True), [False, True, False]
        )

    def test_seed_schedule(self):
        for k, nc in [(70, 50), (49, 33), (0, 0)]:
            x = np.zeros((2, 100))
            y = np.zeros((4, 100))
            y[3] = np.r_[np.full(k, -1.0), np.ones(100 - k)]
            xx, yy, idx = select_seeds(x, y, 0, True, 100, Random(1))
            self.assertEqual(xx.shape[1], nc)

    def test_nonfinite(self):
        self.assertTrue(np.isnan(source_logsum([0, np.nan])))
        self.assertEqual(source_logsum([-np.inf, np.nan]), -np.inf)
        self.assertEqual(source_logsum([np.inf, np.nan]), np.inf)
        self.assertEqual(source_logsum([np.nan, np.nan]), -np.inf)


if __name__ == "__main__":
    unittest.main()
