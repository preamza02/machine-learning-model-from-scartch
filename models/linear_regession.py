import numpy as np
from tqdm import tqdm
from classes.loss_funtion import MSE,MAE
from classes.abstract_class import base_model,
from function.eval_matric import cal_acc

class linear_regession(base_model):
    def __init__(self,
                 loss_function:str = "MSE"
                 ) -> None:
        base_model.__init__(self)
        self.weight:np.ndarray = None
        self.loss:int =self.__set_loss(loss_function)

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

    def __cal_loss(self,y,y_pred) -> int:
        return self.loss.cal(y,y_pred)

    def __back_prob(self,
                    x:np.ndarray,
                    y:np.ndarray,
                    y_pred:np.ndarray,
                    learning_rate:float
                    ) -> None:
        error = y_pred-y

    def predict(self,
                x:np.ndarray,
                ) -> np.ndarray:
        return np.sum(np.dot(self.weight,x),axis=1)

    def train(self,
              x:np.ndarray,
              y:np.ndarray,
              epochs:int = 1,
              batch_size:int = 1,
              learning_rate:float = 1e-3
              ) -> None:
        for round in tqdm(range(epochs)):
            loss = 0
            for round_batch in range(len((x)-1)/batch_size+1):
                batch_x = x[batch_size*round_batch:batch_size*(round_batch+1) - 1]
                batch_y = y[batch_size*round_batch:batch_size*(round_batch+1) - 1]
                y_pred = self.predict(batch_x)
                self.__back_prob(batch_x,batch_y,y_pred,learning_rate)
                loss += self.__cal_loss(y,y_pred)

