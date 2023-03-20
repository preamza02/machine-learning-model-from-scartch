from abc import abstractmethod
from tqdm import tqdm
from .base_class import Base
from ..lossfunctions.diffable_loss import DiffableLoss
import numpy as np


class BaseModel(Base):
    """Abstract base class for models."""

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the given inputs and targets."""
        pass


class BaseRegressionModel(BaseModel):
    """Abstract base class for regression models."""

    def __init__(self, loss_function_name: str) -> None:
        super().__init__()
        self.loss_function: DiffableLoss = self._set_loss(loss_function_name)
        self.weights: np.ndarray = None

    def _init_weights(self, dimensions: tuple) -> np.ndarray:
        """Return randomly initialized weights for the given dimensions."""
        return np.random.rand(dimensions)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the model's predictions for the given inputs."""
        af = np.vectorize(self.activation_function)
        return af(np.sum(np.dot(self.weights, x), axis=1))

    def __calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return the loss between the true and predicted outputs."""
        return self.loss_function.calculate_loss(y_true, y_pred)

    def __backpropagation(self, x: np.ndarray, learning_rate: float) -> None:
        """Update the model's weights using backpropagation."""
        adjustment_rate = self.loss_function.calculate_derivative(x)
        self.weights += learning_rate * adjustment_rate

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, learning_rate: float) -> None:
        """Train the model on the given inputs and targets using mini-batch stochastic gradient descent."""
        if self.weights is None:
            print("Initializing weights...")
            self.weights = self._init_weights(x.shape[1:])
            print("Weight initialization successful!")
        print("Training started...")
        for epoch in tqdm(range(epochs)):
            loss = 0
            for batch in range((len(x)-1)//batch_size+1):
                batch_x = x[batch_size*batch:batch_size*(batch+1) - 1]
                batch_y = y[batch_size*batch:batch_size*(batch+1) - 1]
                y_pred = self.predict(batch_x)
                self.__backpropagation(batch_x, learning_rate)
                loss += self.__calculate_loss(batch_y, y_pred)

    @abstractmethod
    def _set_loss(self, loss_function_name: str) -> DiffableLoss:
        """Return the loss function for the given name."""
        return DiffableLoss()
    
    @abstractmethod
    def activation_function(self, x: float) -> float:
        """Return the activation function applied to x."""
        return x
    

class BaseTreeModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()