import unittest
import numpy as np
from ..base_model import BaseRegressionModel


class TestBaseRegressionModel(unittest.TestCase):
    
    def test_train(self):
        # Generate some random data
        x = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        # Create an instance of the model
        model = BaseRegressionModel(loss_function_name='mse')
        
        # Train the model
        model.train(x, y, epochs=10, batch_size=10, learning_rate=0.01)
        
        # Check that the weights have been updated
        self.assertIsNotNone(model.weights)
        self.assertNotEqual(model.weights.tolist(), np.zeros_like(model.weights).tolist())
        
        # Check that the model can make predictions
        y_pred = model.predict(x)
        self.assertEqual(y_pred.shape, (100,))