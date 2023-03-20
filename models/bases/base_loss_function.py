from .base_class import Base
import numpy as np
from abc import abstractmethod
# from models.eval_matric import cal_acc


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