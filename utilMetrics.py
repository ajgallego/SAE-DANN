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


# ----------------------------------------------------------------------------
def __run_validations(pred, gt):
    assert(isinstance(pred, np.ndarray))
    assert(isinstance(gt, np.ndarray))

    assert(np.issubdtype(pred.dtype.type, np.bool_))
    assert(np.issubdtype(gt.dtype.type, np.bool_))

    assert(len(pred) == len(gt))
    assert(pred.shape[0]==gt.shape[0])
    assert(pred.shape[1]==gt.shape[1])


# ----------------------------------------------------------------------------
def common_metrics(prediction, gt):
    __run_validations(prediction, gt)

    not_prediction = np.logical_not(prediction)
    not_gt = np.logical_not(gt)

    tp = np.logical_and(prediction, gt)
    tn = np.logical_and(not_prediction, not_gt)
    fp = np.logical_and(prediction, not_gt)
    fn = np.logical_and(not_prediction, gt)

    tp = (tp.astype('int32')).sum()
    tn = (tn.astype('int32')).sum()
    fp = (fp.astype('int32')).sum()
    fn = (fn.astype('int32')).sum()

    epsilon = 0.00001
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    fm = 2 * (precision * recall) / (precision + recall + epsilon)
    specificity = tn / (tn + fp + epsilon)

    gt = gt.astype('int32')
    prediction = prediction.astype('int32')

    difference = np.absolute(prediction - gt)
    totalSize = np.prod(gt.shape)
    error = float(difference.sum()) / float(totalSize)

    return {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn,
            'error':error, 'accuracy':accuracy, 'precision':precision,
            'recall':recall, 'fm':fm, 'specificity':specificity}


#------------------------------------------------------------------------------
def run_test(y_pred, y_gt, threshold=.5):
    prediction = y_pred.copy()
    gt = y_gt.copy()

    prediction = (prediction > threshold)
    gt = gt > 0.5

    #gt = np.logical_not(gt)    # Invert the matrix so that the 1 are the positive class
    #prediction = np.logical_not(prediction)

    #cv2.imshow("gt", gt[0].astype(np.uint8)*255)
    #cv2.imshow("prediction", prediction[0].astype(np.uint8)*255)
    #cv2.waitKey(0)

    r = utilMetrics.common_metrics(prediction, gt)
    """r['iou'] = utilMetrics.iou(prediction, gt)
    r['pixa'] = utilMetrics.pixel_accuracy(prediction, gt)
    r['meana'] = utilMetrics.mean_accuracy(prediction, gt)
    r['meaniu'] = utilMetrics.mean_IU(prediction, gt)
    r['frwiu'] = utilMetrics.frequency_weighted_IU(prediction, gt)

    print('TP\tTN\tFP\tFN\tError\tAcc\tPrec\tRecall\tFm\tSpecif.\tIoU\tPixA\tMeanA\tMeanIU\tFrWIU')
    util.print_tabulated([
            r['tp'], r['tn'], r['fp'], r['fn'],
            r['error'], r['accuracy'],
            r['precision'], r['recall'], r['fm'], r['specificity'],
            r['iou'], r['pixa'], r['meana'],
            r['meaniu'], r['frwiu']
    ])"""

    return r

#------------------------------------------------------------------------------
def calculate_best_fm(args_th, y_pred, y_test):
    best_fm = -1
    best_th = -1
    if args_th == -1:
        for i in range(11):
            th = float(i) / 10.0
            print('Threshold:', th)
            results = run_test(y_pred, y_test, threshold=th)
            fm = results['fm']
            if fm > best_fm:
                best_fm = fm
                best_th = th
        print('Best threshold:', best_th)
        print('Best Fm: {:0.4f}'.format(best_fm))

    else:
        results = run_test(y_pred, y_test, threshold=args_th)
        best_fm = results['fm']
        best_th = args_th

    return best_fm, best_th

