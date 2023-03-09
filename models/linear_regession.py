import numpy as np
from tqdm import tqdm
from .classes.loss_funtion import MSE,MAE
from .classes.base_model import base_regession_model
from .function.eval_matric import cal_acc

class linear_regession(base_regession_model):
    def __init__(self,
                 loss_function:str = "MSE"
                 ) -> None:
        super().__init__(self,loss_function)

    def __set_loss(self,
                   loss_function:str,
                   ) -> None:
        match loss_function:
            case "MSE":
                return MSE()
            case "MAE":
                return MAE()
            case _:
                print("Not found loss function set to MSE")
                return MSE()

    def train(self,
              x:np.ndarray,
              y:np.ndarray,
              epochs:int = 1,
              batch_size:int = 1,
              learning_rate:float = 1e-3
              ) -> None:
        super().train(x,y,epochs,batch_size,learning_rate)

