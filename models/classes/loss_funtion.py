from .base_class import Base
import numpy as np
from abc import abstractmethod


class BaseLossFunction(Base):
    def __init__(self) -> None:
        super().__init__()

    def cal(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        af = np.vectorize(self.function)
        return af(y_pred, y)

    @abstractmethod
    def set_name(self) -> str:
        return "base loss function"

    @abstractmethod
    def function(self, pred: float, actual: float) -> float:
        return 


class DiffableLoss(BaseLossFunction):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def set_name(self) -> str:
        return "regression loss"

    @abstractmethod
    def diff(self, x: np.ndarray) -> np.ndarray:
        return x


class MSE(DiffableLoss):
    def __init__(self) -> None:
        super().__init__()

    def set_name(self) -> str:
        return "Mean square error (MSE)"

    def diff(self, x: np.ndarray) -> np.ndarray:
        return super().diff(x)


class MAE(DiffableLoss):
    def __init__(self) -> None:
        super().__init__()

    def set_name(self) -> str:
        return "Mean Absolute error (MAE)"