"""Distribution/iteration/driver separation and shared interface checks."""
import unittest
import numpy as np
from GSS_MIS import GSS_MIS
from SuS import SuS
from SeMCS import SeMCS
from Limit_state_functions.Ex1 import Ex1
from Support_Fun.ScalarExample import ScalarExample
from Support_Fun.CW_aMH import CW_aMH
from Support_Fun.random import Random
from Support_Fun.MIS import MIS
from drive_SuS import drive_SuS, estimate_pf


class FrameworkTests(unittest.TestCase):
    def test_model_is_not_iteration_driver(self):
        model = GSS_MIS(Ex1())
        self.assertFalse(hasattr(model, "RunIte"))
        self.assertFalse(hasattr(model, "EvlPf"))
        engine = SeMCS(CW_aMH(model, Random(3)))
        engine.NumSam = 101
        engine.MaxIte = 1000
        result = engine.RunIte()
        before = [y.copy() for y in result["y"]]
        self.assertEqual(MIS(result).Pf_hat.shape, (2,))
        for old, new in zip(before, result["y"]):
            np.testing.assert_array_equal(old, new)

    def test_sus_rules_are_explicit(self):
        y = np.zeros((4, 101))
        y[3] = np.arange(101) - 50
        pf = ScalarExample(1, 0)
        midpoint = SuS(pf, "physical", "midpoint", False)
        hazen = SuS(pf, "physical", "hazen", False)
        clipped = SuS(pf, "physical", "hazen", True)
        self.assertNotEqual(midpoint.UpdObj(None, y)[1], hazen.UpdObj(None, y)[1])
        self.assertEqual(clipped.UpdObj(None, y)[1], 0)
        self.assertFalse(hasattr(midpoint, "RunIte"))
        self.assertFalse(hasattr(midpoint, "EvlPf"))

    def test_vector_and_predefined_parameters(self):
        def engine():
            value = SeMCS(CW_aMH(GSS_MIS(Ex1()), Random(44)))
            value.NumSam = 100
            value.MaxIte = 1000
            return value
        original = engine().RunIte()
        fixed = engine()
        fixed.G, fixed.H = original["g"], original["h"]
        fixed.NumSam = [100] * 1000
        replay = fixed.RunIte()
        for a, b in zip(original["x"], replay["x"]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(original["Ncal"], replay["Ncal"])

    def test_sus_driver(self):
        result = drive_SuS(1, 100, 2, 12)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["seed"], 12)
        for row in result["records"]:
            self.assertEqual(row["joint_calls"], sum(map(sum, row["layer_joint_calls"])))

    def test_terminal_estimator_uses_zero(self):
        y = np.zeros((4, 3))
        y[3] = [-2, -0.5, 1]
        self.assertEqual(estimate_pf({"y": [y], "g": [np.inf, -1]}, False), 2 / 3)
