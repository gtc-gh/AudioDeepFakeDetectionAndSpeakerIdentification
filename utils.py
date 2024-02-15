import numpy as np
from sklearn.metrics import roc_curve


def equal_error_rate(y_true, y_pred):

    # print("y: ", y_true, y_pred)

    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return EER

