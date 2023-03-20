import unittest
import numpy as np
from unittest.mock import patch
from ..base_loss_function import BaseLossFunction
from unittest.mock import MagicMock

class TestBaseLossFunction(unittest.TestCase):
    def test_calculate_loss(self):
        y = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        blf = BaseLossFunction()
        blf.function = MagicMock(return_value=0.5)
        expected_output = np.array([0.5, 0.5, 0.5])
        self.assertTrue(np.array_equal(blf.calculate_loss(y, y_pred), expected_output))

    def test_set_name(self):
        blf = BaseLossFunction()
        expected_output = "base loss function"
        self.assertEqual(blf._set_name(), expected_output)

    def test_loss_function(self):
        blf = BaseLossFunction()
        blf.function = lambda x: x*x
        y_true = 3.0
        y_pred = 2.0
        expected_output = None
        self.assertEqual(blf.loss_function(y_true, y_pred), expected_output)


if __name__ == '__main__':
    p = patch.multiple(BaseLossFunction, __abstractmethods__=set())
    p.start()
    unittest.main()
    p.stop()