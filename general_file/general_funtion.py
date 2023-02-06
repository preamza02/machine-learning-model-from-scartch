import numpy as np
def ACC(y_true,y_pred,loss_funtion):
    return (y_true==y_pred).sum()/len(y_true)

def TP(y_true,y_pred):
    all_label = np.unique(y_true)

    return None
