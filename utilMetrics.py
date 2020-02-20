# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


# -----------------------------------------------------------------------------
def macroFScore(binarized_imgs, labels_target):
    return f1_score(y_pred=binarized_imgs, y_true=labels_target, average='macro')


# -----------------------------------------------------------------------------
def macroPrecision(binarized_imgs, labels_target):
    return precision_score(y_pred=binarized_imgs, y_true=labels_target, average='macro')


# -----------------------------------------------------------------------------
def macroRecall(binarized_imgs, labels_target):
    return recall_score(y_pred=binarized_imgs, y_true=labels_target, average='macro')


# -----------------------------------------------------------------------------
def __prepare_imgs_to_sklearn(binarized_imgs_cat, labels_target_cat):
    binarized_imgs_2D = np.argmax(binarized_imgs_cat, axis=3)
    labels_target_2D = np.argmax(labels_target_cat, axis=3)

    num_pixels = np.ma.size(binarized_imgs_2D)
    assert(num_pixels == np.ma.size(labels_target_2D))

    binarized_imgs_1D = binarized_imgs_2D.ravel()
    labels_target_1D = labels_target_2D.ravel()

    return binarized_imgs_1D, labels_target_1D


# -----------------------------------------------------------------------------
def calculate_f1(pred, labels):
    pred_1D, labels_1D = __prepare_imgs_to_sklearn(pred, labels)

    f1 = macroFScore(pred_1D, labels_1D)
    precision = macroPrecision(pred_1D, labels_1D)
    recall = macroRecall(pred_1D, labels_1D)

    return precision, recall, f1


