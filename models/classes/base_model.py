from ._base_class import _base,_base_model
from .loss_funtion import regression_loss
from abc import abstractmethod
import numpy as np
from tqdm import tqdm
    
class base_regession_model(_base_model):
    def __init__(self,
                 loss_function:str,
                 ) -> None:
        super().__init__()
        self.lossfunciton:regression_loss = self.__set_loss(loss_function)
        self.weight:np.ndarray = None
    
    def __setname(self) -> str:
        return "base_regession_model"
    
    def __cal_loss(self,
                   y:np.ndarray,
                   y_pred:np.ndarray
                   ) -> float:
        return self.lossfunciton.cal(y,y_pred)
    
    def __init_weight(self,
                     dimension:tuple
                     ) -> np.ndarray:
        return np.random.rand(dimension)
    
    def predict(self,
                x:np.ndarray,
                ) -> np.ndarray:
        af = np.vectorize(self.activate_funtion)
        return af(np.sum(np.dot(self.weight,x),axis=1))
    
    def __back_prob(self,
                    x:np.ndarray,
                    y:np.ndarray,
                    y_pred:np.ndarray,
                    learning_rate:float
                    ) -> None:
        error = y_pred-y

    def activate_funtion(self,x:float) -> float:
        return x

    @abstractmethod
    def train(self,
              x:np.ndarray,
              y:np.ndarray,
              epochs:int,
              batch_size:int,
              learning_rate:float,
              ) -> None:
        if self.weight == None:
            print("start init weight")
            self.weight = self.__init_weight(x.shpae[1:])
            print("init weight succes")
        print("start train")
        for round in tqdm(range(epochs)):
            loss = 0
            for round_batch in range(len((x)-1)/batch_size+1):
                batch_x = x[batch_size*round_batch:batch_size*(round_batch+1) - 1]
                batch_y = y[batch_size*round_batch:batch_size*(round_batch+1) - 1]
                y_pred = self.predict(batch_x)
                self.__back_prob(batch_x,batch_y,y_pred,learning_rate)
                loss += self.__cal_loss(y,y_pred)
    
    @abstractmethod
    def __set_loss(self,
                loss_function:str,
                ) -> regression_loss:
        return regression_loss()
    
