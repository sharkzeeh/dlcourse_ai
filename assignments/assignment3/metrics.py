import numpy as np

def acc(tp : int, tn : int, fp : int, fn: int):
    return (tp + tn) / (tp + fp + fn + tn)

def prec(tp : int, fp: int):
    return tp / (tp + fp)

def rec(tp : int, fn : int):
    return tp / (tp + fn)

def f1_sc(precicion, recall):
    return (2 * precicion * recall) / (precicion + recall)

def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    gt_size = ground_truth.shape[0]

    res = ground_truth[prediction]
    tp = np.sum(res)
    fp = np.sum(np.invert(res))
    fn = np.sum(ground_truth[np.invert(prediction)])
    tn = gt_size - (tp + fp + fn)

    accuracy = acc(tp, tn, fp, fn)
    recall = rec(tp, fn)
    precision = prec(tp, fp)
    f1 = f1_sc(precision, recall)
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    predict_size = prediction.shape[0]
    accuracy = np.sum(ground_truth == prediction) / predict_size
    return accuracy
