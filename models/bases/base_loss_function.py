from .base_class import Base
import numpy as np
from abc import abstractmethod


class BaseLossFunction(Base):
    def __init__(self) -> None:
        super().__init__()

    def calculate_loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        af = np.vectorize(self.function)
        return af(y_pred, y)

    @abstractmethod
    def _set_name(self) -> str:
        return "base loss function"

    @abstractmethod
    def loss_function(self, y_true: float, y_pred: float) -> float:
        pass
    