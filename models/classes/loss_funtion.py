from ._base_class import _base_loss_function
import numpy as np

class regression_loss(_base_loss_function):
    def __init__(self) -> None:
        super().__init__()

        
    def __setname(self) -> str:
        return super().__setname()


class MSE(regression_loss):
    def __init__(self):
        
        self.name = "MSE"


class MAE(regression_loss):
    def __init__(self):
        self.name = "MSE"
    def diff(self) -> np.ndarray:
        return