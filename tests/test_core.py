import unittest
import numpy as np
from Limit_state_functions.benchmark import Benchmark
from Support_Fun.numerics import mround, mstd, quantile, source_logsum
from GSS_MIS import support
from Support_Fun.SedSlt import SedSlt
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
            xx, yy, idx = SedSlt(x, y, 0, True, 100, Random(1))
            self.assertEqual(xx.shape[1], nc)

    def test_nonfinite(self):
        self.assertTrue(np.isnan(source_logsum([0, np.nan])))
        self.assertEqual(source_logsum([-np.inf, np.nan]), -np.inf)
        self.assertEqual(source_logsum([np.inf, np.nan]), np.inf)
        self.assertEqual(source_logsum([np.nan, np.nan]), -np.inf)


class FileAlignmentTests(unittest.TestCase):
    def test_example_and_scalar_interfaces(self):
        from Limit_state_functions.Ex1 import Ex1
        from Limit_state_functions.Ex2 import Ex2
        from Support_Fun.ScalarExample import ScalarExample
        x = np.random.default_rng(41).normal(size=(2, 100))
        for example, cls in ((1, Ex1), (2, Ex2)):
            expected = Benchmark(example).evaluate(x)
            np.testing.assert_array_equal(cls().evaluate(x), expected)
            for component in range(2 * example):
                model = ScalarExample(example, component)
                np.testing.assert_array_equal(
                    model.evaluate(x), expected[component:component + 1]
                )
                self.assertEqual(model.count, 100)
                self.assertEqual(model.joint_count, 100)

    def test_direct_random_order(self):
        from Support_Fun.Direct import Direct
        rng = Random(17)
        expected = np.vstack([rng.randn(1, 30), rng.randn(1, 30)])
        np.testing.assert_array_equal(Direct.sample(Random(17), 30), expected)

    def test_log_helpers(self):
        from Support_Fun.Operations_on_Logarithmic_values import logsum, logmean
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        for axis in (None, 0, 1):
            np.testing.assert_array_equal(logsum(x, axis), source_logsum(x, axis))
            n = x.size if axis is None else x.shape[axis]
            np.testing.assert_array_equal(
                logmean(x, axis), source_logsum(x, axis) - np.log(n)
            )


if __name__ == "__main__":
    unittest.main()
