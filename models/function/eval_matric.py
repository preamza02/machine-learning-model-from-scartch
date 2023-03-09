import numpy as np
def cal_acc(y_true,y_pred,loss_funtion):
    return (y_true==y_pred).sum()/len(y_true)

def cal_tp(y_true,y_pred):
    all_label = np.unique(y_true)

    return None
