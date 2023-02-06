import numpy as np
from tqdm import tqdm
from general_function import ACC
from general_class import MSE,MAE

class linear_regession():
    def __init__(self,loss_function = "MSE"):
        print("initize ")
        self.weight = None
        self.loss =self.__set_loss(loss_function)

    def __set_loss(self,loss_function):
        match loss_function:
            case "MSE":
                return MSE()
            case "MAE":
                return MAE()
            case _:
                print("Not found loss function set to MSE")
                return MSE()

    def __cal_loss(self):
        pass

    def __back_prob(self,x,y):
        pass

    def predict(self,x):
        return np.sum(np.dot(self.weight,x)axis=1)

    def train(self,x,y,epochs = 1,batch_size = 1):
        for round in tqdm(range(epochs)):
            for round_batch in range(len((x)-1)/batch_size+1):
                batch_x = x[batch_size*round_batch:batch_size*(round_batch+1) - 1]
                batch_y = y[batch_size*round_batch:batch_size*(round_batch+1) - 1]
                result = self.predict(batch_x)
