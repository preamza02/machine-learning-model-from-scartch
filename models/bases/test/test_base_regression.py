from ..base_model import BaseRegressionModel
import numpy as np


if __name__ == "__main__":
    import unittest
    from unittest.mock import patch
    class TestBaseRegressionModel(unittest.TestCase):

        def setUp(self):
            self.x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            self.y = np.array([0.5, 0.3, 0.2, 0.9])

        def test_init_weights(self):
            model = BaseRegressionModel("mse")
            dimensions = (2,)
            weights = model._init_weights(dimensions)
            self.assertEqual(weights.shape, dimensions)

        def test_predict(self):
            model = BaseRegressionModel("mse")
            model.weights = np.array([0.1, 0.2])
            y_pred = model.predict(self.x)
            self.assertEqual(y_pred.shape, (4,))

        def test_backpropagation(self):
            model = BaseRegressionModel("mse")
            model.weights = np.array([0.1, 0.2])
            with patch.object(model.loss_function, "calculate_derivative") as mock_loss:
                mock_loss.return_value = np.array([0.3, 0.5, 0.1, 0.7])
                model.__backpropagation(self.x, 0.01)
                expected_weights = np.array([0.103, 0.205])
                np.testing.assert_allclose(model.weights, expected_weights, rtol=1e-3)

        def test_train(self):
            model = BaseRegressionModel("mse")
            with patch.object(model, "predict") as mock_predict:
                mock_predict.return_value = np.array([0.1, 0.2, 0.3, 0.4])
                with patch.object(model, "_init_weights") as mock_weights:
                    mock_weights.return_value = np.array([0.5, 0.1])
                    model.train(self.x, self.y, epochs=10, batch_size=2, learning_rate=0.01)
            self.assertIsNotNone(model.weights)

        def test_set_loss(self):
            model = BaseRegressionModel("mse")
            self.assertIsNotNone(model.loss_function)

        def test_activation_function(self):
            model = BaseRegressionModel("mse")
            x = 0.5
            self.assertEqual(model.activation_function(x), x)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseRegressionModel)
    unittest.TextTestRunner(verbosity=2).run(suite)