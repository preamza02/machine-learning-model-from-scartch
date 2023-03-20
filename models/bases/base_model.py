from abc import abstractmethod
from tqdm import tqdm
from .base_class import Base
from ..lossfunctions.diffable_loss import DiffableLoss,MSE
import numpy as np


class BaseModel(Base):
    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        pass


class BaseRegressionModel(BaseModel):

    def __init__(self, loss_function_name: str) -> None:
        super().__init__()
        self.loss_function: DiffableLoss = self._set_loss(loss_function_name)
        self.weights: np.ndarray = None

    def _init_weights(self, dimensions: tuple) -> np.ndarray:
        return np.random.rand(dimensions)

    def predict(self, x: np.ndarray) -> np.ndarray:
        af = np.vectorize(self.activation_function)
        return af(np.sum(np.dot(self.weights, x), axis=1))

    def __calculate_loss(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray
                         ) -> float:
        return self.loss_function.calculate_loss(y_true, y_pred)

    def __backpropagation(self, 
                          x: np.ndarray, 
                          learning_rate: float
                          ) -> None:
        adjustment_rate = self.loss_function.diff(x)
        self.weights += learning_rate * adjustment_rate

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, learning_rate: float) -> None:
        if self.weights is None:
            print("Initializing weights...")
            self.weights = self._init_weights(x.shape[1:])
            print("Weight initialization successful!")
        print("Training started...")
        for _ in tqdm(range(epochs)):
            loss = 0
            for batch in range((len(x)-1)//batch_size+1):
                batch_x = x[batch_size*batch:batch_size*(batch+1) - 1]
                batch_y = y[batch_size*batch:batch_size*(batch+1) - 1]
                y_pred = self.predict(batch_x)
                self.__backpropagation(batch_x, learning_rate)
                loss += self.__calculate_loss(batch_y, y_pred)

    @abstractmethod
    def _set_loss(self, loss_function_name: str) -> DiffableLoss:
        return MSE()
    
    @abstractmethod
    def activation_function(self, x: float) -> float:
        return x
    

class BaseTreeModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()