"""Joint model-call accounting and completion tests."""

import unittest
import numpy as np
from drive_GSS_MIS import drive_GSS_MIS
from Limit_state_functions.benchmark import Benchmark


class CountTests(unittest.TestCase):
    def test_joint_evaluation(self):
        for example in (1, 2):
            model = Benchmark(example)
            model.evaluate(np.zeros((2, 3)))
            self.assertEqual(model.joint_count, 3)
            self.assertEqual(model.count, 3 * 2 * example)

    def test_single_function(self):
        model = Benchmark(2, pf=1)
        model.evaluate(np.zeros((2, 7)))
        self.assertEqual(model.joint_count, 7)
        self.assertEqual(model.count, 7)

    def test_driver(self):
        for example in (1, 2):
            result = drive_GSS_MIS(example, repeats=2, seed=2026)
            self.assertEqual(result["N"], (440, 410)[example - 1])
            self.assertEqual(result["unfinished"], 0)
            for row in result["records"]:
                self.assertEqual(
                    row["joint_calls"], sum(map(sum, row["layer_joint_calls"]))
                )
                self.assertEqual(len(row["probability"]), example * 2)


if __name__ == "__main__":
    unittest.main()
