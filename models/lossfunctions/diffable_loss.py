from ..bases.base_loss_function import BaseLossFunction
import numpy as np
from abc import abstractmethod

class DiffableLoss(BaseLossFunction):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _set_name(self) -> str:
        return "regression loss"

    @abstractmethod
    def diff(self, x: np.ndarray) -> np.ndarray:
        return x
    
    @abstractmethod
    def function(self, pred: float, actual: float) -> float:
        return 0.0


class MSE(DiffableLoss):
    def __init__(self) -> None:
        super().__init__()

    def _set_name(self) -> str:
        return "Mean square error (MSE)"

    def diff(self, x: np.ndarray) -> np.ndarray:
        return super().diff(x)
    
    def function(self, pred: float, actual: float) -> float:
        return super().function(pred, actual)


class MAE(DiffableLoss):
    def __init__(self) -> None:
        super().__init__()

    def set_name(self) -> str:
        return "Mean Absolute error (MAE)"
    

if __name__ == "__main__":
    print('run')