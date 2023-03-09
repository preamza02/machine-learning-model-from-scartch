from abstract_class import base_loss_function
import numpy as np



class MSE(base_loss_function):
    def __init__(self):
        
        self.name = "MSE"


class MAE(base_loss_function):
    def __init__(self):
        self.name = "MSE"
    def diff(self) -> np.ndarray:
        return