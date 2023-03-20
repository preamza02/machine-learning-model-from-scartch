import numpy as np
import unittest
from unittest.mock import patch

from ..base_model import BaseRegressionModel,BaseModel
from ...lossfunctions.diffable_loss import MSE



class TestBaseRegressionModel(unittest.TestCase):
    
    def test_init(self):
        model = BaseRegressionModel('mse')
        self.assertEqual(model.loss_function.name, 'Mean square error (MSE)')
        self.assertIsNone(model.weights)
    
    def test_weights(self):
        model = BaseRegressionModel('mse')
        self.assertIsNone(model.weights)
        model.weights = np.array([1])
        self.assertTrue(np.array_equal(model.weights, np.array([1])))
    
    def test_init_weights(self):
        model = BaseRegressionModel('mse')
        weights = model._init_weights((1,))
        np.testing.assert_array_less(weights, np.array([1]))
        np.testing.assert_array_less(np.array([0]), weights)
    
    def test_train(self):
        model = BaseRegressionModel('mse')
        x = np.array([1,2,3])
        y = np.array([2,4,6])
        epochs = 1
        batch_size = 2
        learning_rate = 0.01
        with patch.object(model, '_init_weights', return_value=np.array([0.5])):
            model.train(x, y, epochs, batch_size, learning_rate)
        
    def test_calculate_loss(self):
        model = BaseRegressionModel('mse')
        y_true = np.array([1,2])
        y_pred = np.array([1.5,2.5])
        loss = model._BaseRegressionModel__calculate_loss(y_true, y_pred)
        expected_loss = (0.5*(y_true-y_pred)**2).mean()
        self.assertAlmostEqual(loss, expected_loss, places=10)
        
    def test_predict(self):
        model = BaseRegressionModel('mse')
        model.weights = np.array([3,2,1])
        x = np.array([[1,2,3],[4,5,6]])
        pred = model.predict(x)
        expected_pred = np.array([10, 28])
        np.testing.assert_array_equal(pred, expected_pred)
        
    def test_set_loss(self):
        model = BaseRegressionModel('mse')
        loss = model._set_loss('mse')
        self.assertIsInstance(loss, MSE)
        
    def test_activation_function(self):
        model = BaseRegressionModel('mse')
        self.assertEqual(model.activation_function(5), 5)


if __name__ == '__main__':

    pmse = patch.multiple(MSE, __abstractmethods__=set())
    p = patch.multiple(BaseRegressionModel, __abstractmethods__=set())
    pmse.start()
    p.start()
    unittest.main()
    p.stop()
    pmse.stop()