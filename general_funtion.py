import numpy as np
def ACC(y_true:np.ndarray,y_pred:np.ndarray):
    return (y_true==y_pred).sum()/len(y_true)

