import unittest
from unittest.mock import patch
from ..base_model import BaseModel
import numpy as np

class TestBaseModel(unittest.TestCase):
    def test_train(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        base_model = BaseModel()
        self.assertEqual(base_model.train(x, y),None)

if __name__ == '__main__':
    p = patch.multiple(BaseModel, __abstractmethods__=set())
    p.start()
    unittest.main()
    p.stop()