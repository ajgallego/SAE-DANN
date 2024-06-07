# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import cv2
import numpy as np
import util
import utilConst
import utilDataGenerator
import utilMetrics
from collections import Counter

import math


# -----------------------------------------------------------------------------
def load_array_of_files(basepath, folders, truncate=False):
    X = []
    for folder in folders:
        full_path = os.path.join(basepath, folder)
        array_of_files = util.list_files(full_path, ext='png')

        for fname_x in array_of_files:
            X.append(fname_x)

    if truncate:
        X = X[:10]

    return np.asarray(X)


# ----------------------------------------------------------------------------
def load_folds_names(dbname):
    assert dbname in utilConst.ARRAY_DBS

    train_folds = []
    test_folds = []

    DIBCO = [    ['Dibco/2009/handwritten_GR', 'Dibco/2009/printed_GR'],
                 ['Dibco/2010/handwritten_GR'],
                 ['Dibco/2011/handwritten_GR', 'Dibco/2011/printed_GR'],
                 ['Dibco/2012/handwritten_GR'],
                 ['Dibco/2013/handwritten_GR', 'Dibco/2013/printed_GR'],
                 ['Dibco/2014/handwritten_GR'],
                 ['Dibco/2016/handwritten_GR']     ]

    DIBCO_synthetic_inv_col = [
                 ['synthetic/inv_col/Dibco/2009/handwritten_GR', 'synthetic/inv_col/Dibco/2009/printed_GR'],
                 ['synthetic/inv_col/Dibco/2010/handwritten_GR'],
                 ['synthetic/inv_col/Dibco/2011/handwritten_GR', 'synthetic/inv_col/Dibco/2011/printed_GR'],
                 ['synthetic/inv_col/Dibco/2012/handwritten_GR'],
                 ['synthetic/inv_col/Dibco/2013/handwritten_GR', 'synthetic/inv_col/Dibco/2013/printed_GR'],
                 ['synthetic/inv_col/Dibco/2014/handwritten_GR'],
                 ['synthetic/inv_col/Dibco/2016/handwritten_GR']     ]

    PALM_train = [ ['Palm/Challenge-1-ForTrain/gt1_GR'], ['Palm/Challenge-1-ForTrain/gt2_GR'] ]
    PALM_test = [ ['Palm/Challenge-1-ForTest/gt1_GR'], ['Palm/Challenge-1-ForTest/gt2_GR'] ]

    PALM_train_synthetic_inv_col = [ ['synthetic/inv_col/Palm/Challenge-1-ForTrain/gt1_GR'], ['synthetic/inv_col/Palm/Challenge-1-ForTrain/gt2_GR'] ]
    PALM_test_synthetic_inv_col = [ ['synthetic/inv_col/Palm/Challenge-1-ForTest/gt1_GR'], ['synthetic/inv_col/Palm/Challenge-1-ForTest/gt2_GR'] ]

    PHI_train = ['PHI/train/phi_GR']
    PHI_test = ['PHI/test/phi_GR']

    PHI_train_synthetic_inv_col = ['synthetic/inv_col/PHI/train/phi_GR']
    PHI_test_synthetic_inv_col = ['synthetic/inv_col/PHI/test/phi_GR']

    EINSIELDELN_train = ['Einsieldeln/train/ein_GR']
    EINSIELDELN_test = ['Einsieldeln/test/ein_GR']

    EINSIELDELN_train_synthetic_inv_col = ['synthetic/inv_col/Einsieldeln/train/ein_GR']
    EINSIELDELN_test_synthetic_inv_col = ['synthetic/inv_col/Einsieldeln/test/ein_GR']

    SALZINNES_train = ['Salzinnes/train/sal_GR']
    SALZINNES_test = ['Salzinnes/test/sal_GR']

    SALZINNES_train_synthetic_overexposure10 = ['synthetic/overexposure_g10/Salzinnes/train/sal_GR']
    SALZINNES_test_synthetic_overexposure10 = ['synthetic/overexposure_g10/Salzinnes/test/sal_GR']

    SALZINNES_train_synthetic_overexposure0_4 = ['synthetic/overexposure_g0.4/Salzinnes/train/sal_GR']
    SALZINNES_test_synthetic_overexposure0_4 = ['synthetic/overexposure_g0.4/Salzinnes/test/sal_GR']

    SALZINNES_train_synthetic_blur30x30 = ['synthetic/blur_30x30/Salzinnes/train/sal_GR']
    SALZINNES_test_synthetic_blur30x30 = ['synthetic/blur_30x30/Salzinnes/test/sal_GR']

    SALZINNES_train_synthetic_inv_col = ['synthetic/inv_col/Salzinnes/train/sal_GR']
    SALZINNES_test_synthetic_inv_col = ['synthetic/inv_col/Salzinnes/test/sal_GR']

    VOYNICH_test = ['Voynich/voy_GR']
    VOYNICH_test_synthetic_inv_col = ['synthetic/inv_col/Voynich/voy_GR']

    BDI_train = ['BDI/train/bdi11_GR']
    BDI_test = ['BDI/test/bdi11_GR']

    BDI_train_synthetic_inv_col = ['synthetic/inv_col/BDI/train/bdi11_GR']
    BDI_test_synthetic_inv_col = ['synthetic/inv_col/BDI/test/bdi11_GR']

    LRDE_DBD_train = ['LRDE_DBD/train/GR']
    LRDE_DBD_test  = ['LRDE_DBD/test/GR']


    if dbname == 'dibco2016':
        test_folds = DIBCO[6]
        DIBCO.pop(6)
        train_folds = [val for sublist in DIBCO for val in sublist]
    elif dbname == 'dibco2016-ic':
        test_folds = DIBCO_synthetic_inv_col[6]
        DIBCO_synthetic_inv_col.pop(6)
        train_folds = [val for sublist in DIBCO_synthetic_inv_col for val in sublist]
    elif dbname == 'dibco2014':
        test_folds = DIBCO[5]
        DIBCO.pop(5)
        train_folds = [val for sublist in DIBCO for val in sublist]
    elif dbname == 'dibco2014-ic':
        test_folds = DIBCO_synthetic_inv_col[5]
        DIBCO_synthetic_inv_col.pop(5)
        train_folds = [val for sublist in DIBCO_synthetic_inv_col for val in sublist]
    elif dbname == 'palm0':
        train_folds = PALM_train[0]
        test_folds = PALM_test[0]
    elif dbname == 'palm0-ic':
        train_folds = PALM_train_synthetic_inv_col[0]
        test_folds = PALM_test_synthetic_inv_col[0]
    elif dbname == 'palm1':
        train_folds = PALM_train[1]
        test_folds = PALM_test[1]
    elif dbname == 'palm1-ic':
        train_folds = PALM_train_synthetic_inv_col[1]
        test_folds = PALM_test_synthetic_inv_col[1]
    elif dbname == 'phi':
        train_folds = PHI_train
        test_folds = PHI_test
    elif dbname == 'phi-ic':
        train_folds = PHI_train_synthetic_inv_col
        test_folds = PHI_test_synthetic_inv_col
    elif dbname == 'ein':
        train_folds = EINSIELDELN_train
        test_folds = EINSIELDELN_test
    elif dbname == 'ein-ic':
        train_folds = EINSIELDELN_train_synthetic_inv_col
        test_folds = EINSIELDELN_test_synthetic_inv_col
    elif dbname == 'sal':
        train_folds = SALZINNES_train
        test_folds = SALZINNES_test
    elif dbname == 'sal-ic':
        train_folds = SALZINNES_train_synthetic_inv_col
        test_folds = SALZINNES_test_synthetic_inv_col
    elif dbname == 'sal-oe10':
        train_folds = SALZINNES_train_synthetic_overexposure10
        test_folds = SALZINNES_test_synthetic_overexposure10
    elif dbname == 'sal-oe0.4':
        train_folds = SALZINNES_train_synthetic_overexposure0_4
        test_folds = SALZINNES_test_synthetic_overexposure0_4
    elif dbname == 'sal-blur30x30':
        train_folds = SALZINNES_train_synthetic_blur30x30
        test_folds = SALZINNES_test_synthetic_blur30x30
    elif dbname == 'lrde':
        train_folds = LRDE_DBD_train
        test_folds  = LRDE_DBD_test
    elif dbname == 'voy':
        train_folds = [val for sublist in DIBCO for val in sublist]
        test_folds = VOYNICH_test
    elif dbname == 'voy-ic':
        train_folds = [val for sublist in DIBCO_synthetic_inv_col for val in sublist]
        test_folds = VOYNICH_test_synthetic_inv_col
    elif dbname == 'bdi':
        train_folds = BDI_train
        test_folds = BDI_test
    elif dbname == 'bdi-ic':
        train_folds = BDI_train_synthetic_inv_col
        test_folds = BDI_test_synthetic_inv_col
    elif dbname == 'all':
        test_folds = [DIBCO[5], DIBCO[6]]
        test_folds.append(PALM_test[0])
        test_folds.append(PALM_test[1])
        test_folds.append(PHI_test)
        test_folds.append(EINSIELDELN_test)
        test_folds.append(SALZINNES_test)
        test_folds.append(LRDE_DBD_test)

        DIBCO.pop(6)
        DIBCO.pop(5)
        train_folds = [[val for sublist in DIBCO for val in sublist]]
        train_folds.append(PALM_train[0])
        train_folds.append(PALM_train[1])
        train_folds.append(PHI_train)
        train_folds.append(EINSIELDELN_train)
        train_folds.append(SALZINNES_train)
        train_folds.append(LRDE_DBD_train)

        test_folds = [val for sublist in test_folds for val in sublist]  # transform to flat lists
        train_folds = [val for sublist in train_folds for val in sublist]
    elif dbname == 'all-ic':
        test_folds = [DIBCO_synthetic_inv_col[5], DIBCO_synthetic_inv_col[6]]
        test_folds.append(PALM_test_synthetic_inv_col[0])
        test_folds.append(PALM_test_synthetic_inv_col[1])
        test_folds.append(PHI_test_synthetic_inv_col)
        test_folds.append(EINSIELDELN_test_synthetic_inv_col)
        test_folds.append(SALZINNES_test_synthetic_inv_col)

        DIBCO_synthetic_inv_col.pop(6)
        DIBCO_synthetic_inv_col.pop(5)
        train_folds = [[val for sublist in DIBCO_synthetic_inv_col for val in sublist]]
        train_folds.append(PALM_train_synthetic_inv_col[0])
        train_folds.append(PALM_train_synthetic_inv_col[1])
        train_folds.append(PHI_train_synthetic_inv_col)
        train_folds.append(EINSIELDELN_train_synthetic_inv_col)
        train_folds.append(SALZINNES_train_synthetic_inv_col)

        test_folds = [val for sublist in test_folds for val in sublist]  # transform to flat lists
        train_folds = [val for sublist in train_folds for val in sublist]
    else:
        raise Exception('Unknown database name')

    return train_folds, test_folds


#------------------------------------------------------------------------------
def __calculate_img_diff(img_pr, img_y):
    assert img_pr is not None and img_y is not None
    assert img_pr.shape == img_y.shape

    img_diff = np.zeros((img_pr.shape[0], img_pr.shape[1], 3), np.uint8)

    for f in xrange(img_pr.shape[0]):
        for c in xrange(img_pr.shape[1]):
            if img_pr[f,c] == img_y[f,c]:
                img_diff[f,c] = (0,0,0) if img_y[f,c] == 0 else (255,255,255)
            else:
                img_diff[f,c] = (0,0,255) if img_y[f,c] == 0 else (255,0,0)

    return img_diff


def getPrecision(num_decimal):
    precision = 1.
    for _ in range(num_decimal):
        precision /= 10.

    return precision

def getHistogram(image, num_decimal):

    tuple_prediction = tuple(image.reshape(1,-1)[0])

    if num_decimal is not None:
        tuple_prediction_round = []
        for num in tuple_prediction:
            if num > 0.01:
                tuple_prediction_round.append(round(num, num_decimal))
            
        #tuple_prediction_round = [round(num, num_decimal) for num in tuple_prediction]
        tuple_prediction = tuple_prediction_round

        precision = getPrecision(num_decimal)
        
        value = 0.
        value = round(value, num_decimal)
        while value <= 1:
            tuple_prediction.append(value)
            value += precision
            value = round(value, num_decimal)
    
    histogram_prediction = Counter(tuple_prediction)
    return histogram_prediction

# ----------------------------------------------------------------------------
def getHistograms(model, array_files_to_save, config, threshold = 0.5, num_decimal=None):

    print('Calculating histogram...')

    list_histograms = []

    array_files = load_array_of_files(config.path, array_files_to_save)

    for fname in array_files:
        print('Processing image', fname)

        fname_gt = fname.replace(utilConst.X_SUFIX, utilConst.Y_SUFIX)
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(fname_gt, cv2.IMREAD_GRAYSCALE)

        rows = img.shape[0]
        cols = img.shape[1]
        if img.shape[0] < config.window or img.shape[1] < config.window:
            new_rows = config.window if img.shape[0] < config.window else img.shape[0]
            new_cols = config.window if img.shape[1] < config.window else img.shape[1]
            img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

        #cv2.imshow("img", img)
        #cv2.waitKey(0)

        finalImg = np.zeros(img.shape, dtype=float)
        
        for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=config.step-5, windowSize=(config.window, config.window)):
            if window.shape[0] != config.window or window.shape[1] != config.window:
                continue

            roi = img[y:(y + config.window), x:(x + config.window)].copy()

            #cv2.imshow("roi", roi)
            #cv2.waitKey(0)

            roi = roi.reshape(1, config.window, config.window, 1)
            roi = roi.astype('float32')
            norm_type = '255'
            roi = utilDataGenerator.normalize_data( roi, norm_type )

            prediction = model.predict(roi)

            prediction = prediction[:,2:prediction.shape[1]-2,2:prediction.shape[2]-2,:]
            finalImg[y+2:(y + config.window-2), x+2:(x + config.window-2)] = prediction[0].reshape(config.window-4, config.window-4)
           
            #cv2.imshow("finalImg", (1 - finalImg.astype('uint8')) * 255 )
            #cv2.waitKey(0)
        
        finalImg_bin = (finalImg >= threshold)
        finalImg_bin = (1 - finalImg_bin.astype('uint8'))

        import ntpath
        filename = ntpath.basename(fname)
        filename_out = filename.replace(".", "_"+str(config.type) + ".")

        pathdir_outimage = "OUTPUT/probs/" + str(config.db1) +"-"+ str(config.db2) + "/"

        util.mkdirp( os.path.dirname(pathdir_outimage) )
        cv2.imwrite(pathdir_outimage + str(filename_out), finalImg_bin*255)
        cv2.imwrite(pathdir_outimage + str(filename), 255-img*255)
        
        histogram_prediction = getHistogram(finalImg, num_decimal)
        list_histograms.append(histogram_prediction)

        out_histogram_filename = fname.replace(config.path, 'OUTPUT/histogram')

        out_histogram_filename = out_histogram_filename.replace(utilConst.X_SUFIX+'/', '/'+config.modelpath + '/')
        out_histogram_filename = out_histogram_filename.replace(utilConst.WEIGHTS_DANN_FOLDERNAME+'/', '')
        out_histogram_filename = out_histogram_filename.replace(utilConst.WEIGHTS_CNN_FOLDERNAME+'/', '')
        out_histogram_filename = out_histogram_filename.replace(utilConst.LOGS_DANN_FOLDERNAME+'/', '')
        out_histogram_filename = out_histogram_filename.replace(utilConst.LOGS_CNN_FOLDERNAME+'/', '')

        out_histogram_filename = out_histogram_filename.replace('.h5', '/OUT_PR').replace('.npy', '/OUT_PR')

        out_histogram_filename = str(out_histogram_filename.replace(".png", ".txt"))
        print(' - Saving predicted image to:', out_histogram_filename)
        util.mkdirp( os.path.dirname(out_histogram_filename) )

        items_histogram = sorted(histogram_prediction.items())
        str_prob = ""
        str_value = ""

        for prob, value in items_histogram:
            str_prob += str(prob) + "\t"
            str_value += str(value-1) + "\t"

        str_histogram = str_prob + "\n" + str_value

        #tuple_prediction_round = [str_prob for prob, value in items_histogram]

        saveString(str_histogram, out_histogram_filename, True)

    return list_histograms
        

def getNormalizedHistogram(histogram):
    print ("---------------------")
    print (histogram)
    values = getHistogramValuesSorted(histogram)
    print(values)
    total = np.sum(values)
    values = [v/float(total) for v in values]
    print (values)
    print ("---------------------")
    return values

def getHistogramValuesSorted(histogram):
    values = []
    items_histogram = sorted(histogram.items())
    
    for prob, value in items_histogram:
        values.append(value)    
    return values



def getHistogramBins(sample_image, num_decimal):
    tuple_sample = tuple(sample_image.reshape(1,-1)[0])

    if num_decimal is not None:
        tuple_sample_round = []
        for num in tuple_sample:
            if num > 0.01:
                tuple_sample_round.append(round(num, num_decimal))
            
        tuple_sample = tuple_sample_round

        precision = 1.
        for i in range(num_decimal):
            precision /= 10.

        value = 0.
        value = round(value, num_decimal)
        while value <= 1:
            tuple_sample.append(value)
            value += precision
            value = round(value, num_decimal)
    
    histogram_prediction = Counter(tuple_sample)

    return histogram_prediction


def getHistogramDomain(array_files, model, config, num_decimal=None):

    histogram_domain = None

    array_files = load_array_of_files(config.path, array_files)
    for fname in array_files:
        print('Processing image', fname)

        #fname_gt = fname.replace(utilConst.X_SUFIX, utilConst.Y_SUFIX)
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        if img.shape[0] < config.window or img.shape[1] < config.window:
            new_rows = config.window if img.shape[0] < config.window else img.shape[0]
            new_cols = config.window if img.shape[1] < config.window else img.shape[1]
            img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)
            
        finalImg = np.zeros(img.shape, dtype=float)
        
        for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=config.step-5, windowSize=(config.window, config.window)):
            if window.shape[0] != config.window or window.shape[1] != config.window:
                continue

            roi = img[y:(y + config.window), x:(x + config.window)].copy()

            roi = roi.reshape(1, config.window, config.window, 1)
            roi = roi.astype('float32')
            norm_type = '255'
            roi = utilDataGenerator.normalize_data( roi, norm_type )

            prediction = model.predict(roi)
            prediction = prediction[:,2:prediction.shape[1]-2,2:prediction.shape[2]-2,:]

            sample_prediction = prediction[0].reshape(config.window-4, config.window-4)
            finalImg[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction
           
        histogram_domain_fname = getHistogram(finalImg, num_decimal)
        print(str(histogram_domain_fname))

        if histogram_domain is None:
            histogram_domain = histogram_domain_fname.copy()
        else:
            histogram_domain = histogram_domain + histogram_domain_fname

        print(str(histogram_domain))

    return histogram_domain



def predictSAE(
                        model_cnn, 
                        config, 
                        target_test_folds, 
                        source_best_th_cnn, 
                        threshold_correl_pearson, 
                        num_decimal=None):

    print('Calculating SAE...')

    array_files = load_array_of_files(config.path, array_files_to_save)

    list_target_best_fm_cnn = []
    list_target_best_fm_cnn_inv = []

    for fname in array_files:
        print('Processing image', fname)

        fname_gt = fname.replace(utilConst.X_SUFIX, utilConst.Y_SUFIX)
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(fname_gt, cv2.IMREAD_GRAYSCALE)

        if img.shape[0] < config.window or img.shape[1] < config.window:
            new_rows = config.window if img.shape[0] < config.window else img.shape[0]
            new_cols = config.window if img.shape[1] < config.window else img.shape[1]
            img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

        finalImg = np.zeros(img.shape, dtype=float)
        finalImg_bin = np.zeros(img.shape, dtype=float)

        finalImg_cnn = np.zeros(img.shape, dtype=float)
        finalImg_bin_cnn = np.zeros(img.shape, dtype=float)
        
        for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=config.window-5, windowSize=(config.window, config.window)):
            if window.shape[0] != config.window or window.shape[1] != config.window:
                continue

            roi = img[y:(y + config.window), x:(x + config.window)].copy()

            roi = roi.reshape(1, config.window, config.window, 1)
            roi = roi.astype('float32')
            norm_type = '255'
            roi = utilDataGenerator.normalize_data( roi, norm_type )

            prediction_cnn = model_cnn.predict(roi)
            prediction_cnn = prediction_cnn[:,2:prediction_cnn.shape[1]-2,2:prediction_cnn.shape[2]-2,:]
            sample_prediction_cnn = prediction_cnn[0].reshape(config.window-4, config.window-4)
            sample_prediction = sample_prediction_cnn
            
            finalImg[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction
            finalImg_bin[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction < threshold)

            finalImg_cnn[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction_cnn
            finalImg_bin_cnn[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction_cnn < threshold_cnn)

        
        import ntpath
        filename = ntpath.basename(fname)
        filename_out = filename.replace(".", "_"+str(config.type) + ".")
        filename_out_cnn = filename.replace(".", "_"+str(config.type) + "_cnn.")
        
        pathdir_outimage = "OUTPUT/sae/probs/" + str(config.db1) +"-"+ str(config.db2) + "/"

        util.mkdirp( os.path.dirname(pathdir_outimage) )
        cv2.imwrite(pathdir_outimage + str(filename_out), finalImg_bin*255)
        cv2.imwrite(pathdir_outimage + str(filename), 255-img*255)
        cv2.imwrite(pathdir_outimage + str(filename_out_cnn), finalImg_bin_cnn*255)

        finalImg_bin = (finalImg_bin>source_best_th_cnn)
        finalImg_bin_cnn = (finalImg_bin_cnn>source_best_th_cnn)
        gt = (gt > 0.5)

        finalImg_bin_inv = (finalImg_bin<=source_best_th_cnn)
        finalImg_bin_cnn_inv = (finalImg_bin_cnn<=source_best_th_cnn)
        gt_inv = (gt <= 0.5)

        print ("SAE:")
        target_best_fm_cnn, _, _, _ = utilMetrics.calculate_best_fm(finalImg_bin_cnn, gt, None)
        
        list_target_best_fm_cnn.append(target_best_fm_cnn)

    print ("F1 at page-level")
    
    str_f1 = "SAE:\t"
    str_f1 += str(list_target_best_fm_cnn)
    
    avg_cnn = np.average(list_target_best_fm_cnn)

    str_f1 +=  ("\nAVERAGE SAE\n")
    str_f1 +=  (str(avg_cnn))

    print(str_f1)
    pathdir_F1 = "OUTPUT/sae/probs/" + str(config.db1) +"-"+ str(config.db2) + "/" + "f1.txt"
    saveString(str_f1, pathdir_F1, True)
    
    return avg_cnn


def predictModelAtFullPage(
                        model, 
                        config, 
                        model_type,
                        array_files, 
                        source_best_th, 
                        num_decimal=None):
    print('Calculating ' + str(model_type) + '...')

    array_files = load_array_of_files(config.path, array_files)

    list_target_best_fm = []
    list_target_results = []

    for fname in array_files:
        print('Processing image', fname)

        fname_gt = fname.replace(utilConst.X_SUFIX, utilConst.Y_SUFIX)
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(fname_gt, cv2.IMREAD_GRAYSCALE)

        if img.shape[0] < config.window or img.shape[1] < config.window:
            new_rows = config.window if img.shape[0] < config.window else img.shape[0]
            new_cols = config.window if img.shape[1] < config.window else img.shape[1]
            img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

        finalImg_prob = np.zeros(img.shape, dtype=float)
        finalImg_bin = np.zeros(img.shape, dtype=float)
        
        for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=config.window-5, windowSize=(config.window, config.window)):
            if window.shape[0] != config.window or window.shape[1] != config.window:
                continue

            roi = img[y:(y + config.window), x:(x + config.window)].copy()

            roi = roi.reshape(1, config.window, config.window, 1)
            roi = roi.astype('float32')
            norm_type = '255'
            roi = utilDataGenerator.normalize_data( roi, norm_type )

            prediction_cnn = model.predict(roi)
            prediction_cnn = prediction_cnn[:,2:prediction_cnn.shape[1]-2,2:prediction_cnn.shape[2]-2,:]
            sample_prediction_cnn = prediction_cnn[0].reshape(config.window-4, config.window-4)
            
            finalImg_prob[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction_cnn
            finalImg_bin[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction_cnn < source_best_th)

        
        import ntpath
        filename = ntpath.basename(fname)
        filename_out_prob = filename.replace(".", "_"+str(config.type) + ".")
        filename_out_cnn = filename.replace(".", "_"+str(config.type) + "_"+ str(model_type) + ".")
        
        pathdir_outimage = "OUTPUT/" + str(model_type) + "/probs/" + str(config.db1) +"-"+ str(config.db2) + "/"

        util.mkdirp( os.path.dirname(pathdir_outimage) )
        cv2.imwrite(pathdir_outimage + str(filename_out_prob), finalImg_prob*255)
        cv2.imwrite(pathdir_outimage + str(filename), 255-img*255)
        cv2.imwrite(pathdir_outimage + str(filename_out_cnn), finalImg_bin*255)

        finalImg_bin = (finalImg_bin>source_best_th)
        finalImg_bin = (finalImg_bin>source_best_th)
        gt = (gt > 0.5)

        print ("F1 at sample-level:")
        target_best_fm_cnn, _, target_precision_cnn, target_recall_cnn = utilMetrics.calculate_best_fm(finalImg_bin, gt, None)
        
        list_target_best_fm.append(target_best_fm_cnn)
        list_target_results.append( (target_best_fm_cnn, target_precision_cnn, target_recall_cnn ))

    str_f1 = "F1 at page-level:\t"
    str_f1 += str(list_target_results)
    
    avg_cnn = np.average(list_target_best_fm)

    str_f1 +=  ("\nAVERAGE " + str(model_type) +  "\n")
    str_f1 +=  (str(avg_cnn))

    print(str_f1)
    pathdir_F1 = "OUTPUT/" + str(model_type) + "/probs/" + str(config.db1) +"-"+ str(config.db2) + "/" + "f1.txt"
    saveString(str_f1, pathdir_F1, True)
    
    return avg_cnn


def histogram_intersection(h1, h2):
    assert(len(h1) == len(h2))
    sm = 0
    for i in range(len(h1)):
        sm += min(h1[i], h2[i])
    return sm



def log2(x):
    return math.log(x)/math.log(2)

# calculate the kl divergence (measured in bits)
def kl_divergence_bits(p, q):
    cte = 0.01
    p2 = [p[i] + cte for i in range(len(p))]
    q2 = [q[i] + cte for i in range(len(q))]
    return sum([p2[i] * log2(p2[i]/q2[i]) for i in range(len(p2))])

# calculate the kl divergence (measured in nats)
def kl_divergence_nats(p, q):
    cte = 0.01
    p2 = [p[i] + cte for i in range(len(p))]
    q2 = [q[i] + cte for i in range(len(q))]
    return sum([p2[i] * math.log(p2[i]/q2[i]) for i in range(len(p2))])

# calculate the js divergence (measure in bits)
def js_divergence_bits(p, q):
    m = [0.5 * (p[i] + q[i]) for i in range(len(q))]
    return 0.5 * kl_divergence_bits(p, m) + 0.5 * kl_divergence_bits(q, m)

# calculate the js divergence (measured in nats)
def js_divergence_nats(p, q):
    m = [0.5 * (p[i] + q[i]) for i in range(len(q))]
    return 0.5 * kl_divergence_nats(p, m) + 0.5 * kl_divergence_nats(q, m)


def pearson_correlation(p, q):
    return np.corrcoef(p, q)[0, 1]

def get_correlation_metric(correlation_type, p, q):
    if correlation_type == "pearson":
        return pearson_correlation(p, q)
    elif correlation_type == "kl-nats":
        return kl_divergence_nats(p, q)
    elif correlation_type == "js-nats":
        return js_divergence_nats(p, q)
    elif correlation_type == "kl-bits":
        return kl_divergence_bits(p, q)
    elif correlation_type == "js-bits":
        return js_divergence_bits(p, q)
    elif correlation_type == "hist-intersection":
        return histogram_intersection(p, q)

def get_all_correlation_metrics(p, q):
    pearson = get_correlation_metric("pearson", p, q)
    
    kl_nats = get_correlation_metric("kl-nats", p, q)
    kl_bits = get_correlation_metric("kl-bits", p, q)
    
    js_nats = get_correlation_metric("js-nats", p, q)
    js_bits = get_correlation_metric("js-bits", p, q)

    hist_inter = get_correlation_metric("hist-intersection", p, q)

    print ("Pearson;KL (nats);KL (bits);JS (nats);JS (bits);Histogram intersection")
    print (str(str(pearson) + ";"\
            + str(kl_nats) + ";"\
            + str(kl_bits) + ";"\
            + str(js_nats) + ";"\
            + str(js_bits) + ";"\
            + str(hist_inter)).replace(".", ","))



def predictAutoDANN_AtSampleLevel(
                        model_dann, 
                        model_cnn,
                        config, 
                        x_test_samples,
                        y_test_samples,
                        source_best_th_cnn, 
                        source_best_th_dann,
                        correlation_type, threshold_correl,
                        histogram_source_cnn,
                        num_decimal):
    predicts_auto = []
    predicts_auto_ideal = []

    #Histogram for source
    list_histogram_source_cnn = histogram_source_cnn.values()
    number_pixels_source = sum(list_histogram_source_cnn)
    normalized_list_histogram_source_cnn = [number / float(number_pixels_source) for number in list_histogram_source_cnn]
    
    count_cnn = 0
    count_dann = 0

    count_cnn_ideal = 0
    count_dann_ideal = 0

    idx_sample = 0
    for idx_sample in range(len(x_test_samples)):
        x_test_sample = x_test_samples[idx_sample]
        y_test_sample = y_test_samples[idx_sample]
        
        list_x_test_sample = list()
        list_x_test_sample.append(x_test_sample)
        list_x_test_sample_array = np.asarray(list_x_test_sample)

        prediction_cnn = model_cnn.predict(list_x_test_sample_array)
        sample_prediction_cnn = prediction_cnn[0].reshape(config.window, config.window)

        #Histogram for target
        histogram_prediction_cnn = getHistogramBins(prediction_cnn, num_decimal)
        list_histogram_prediction_cnn = histogram_prediction_cnn.values()
        number_pixels_target = sum(list_histogram_prediction_cnn)

        normalized_list_histogram_prediction_cnn = [number / float(number_pixels_target) for number in list_histogram_prediction_cnn]

        correlation = get_correlation_metric(correlation_type, normalized_list_histogram_prediction_cnn, normalized_list_histogram_source_cnn)
        #get_all_correlation_metrics(normalized_list_histogram_prediction_cnn, normalized_list_histogram_source_cnn)

        prediction_dann = model_dann.predict(list_x_test_sample_array)
        sample_prediction_dann = prediction_dann[0].reshape(config.window, config.window)
        
        if correlation > threshold_correl:
            #SAE
            threshold = source_best_th_cnn
            sample_prediction = sample_prediction_cnn
            count_cnn += 1

        else:
            #DANN
            threshold = source_best_th_dann
            sample_prediction = sample_prediction_dann
            count_dann += 1

        predicts_auto.append(sample_prediction > threshold)


        sample_prediction_cnn_th = sample_prediction_cnn > source_best_th_cnn
        sample_prediction_dann_th = sample_prediction_dann > source_best_th_dann

        y_test_sample = y_test_sample[:,:,0].reshape(config.window, config.window)
        
        #cv2.imwrite("prueba/cnn.png", sample_prediction_cnn_th*255)
        #cv2.imwrite("prueba/dann.png", sample_prediction_dann_th*255)
        #cv2.imwrite("prueba/gt.png", (y_test_sample>0.5)*255)
        
        #print (sample_prediction_cnn_th.shape)
        #print(sample_prediction_dann_th.shape)
        #print(y_test_sample.shape)
         

        target_best_fm_cnn, _, target_precision_cnn, target_recall_cnn = utilMetrics.calculate_best_fm(sample_prediction_cnn_th, y_test_sample > 0.5, None, False)
        target_best_fm_dann, _, target_precision_dann, target_recall_dann = utilMetrics.calculate_best_fm(sample_prediction_dann_th, y_test_sample > 0.5, None, False)

        if target_best_fm_cnn > target_best_fm_dann:
            best_sample_prediction_th = sample_prediction_cnn_th
            count_cnn_ideal += 1
        else:
            best_sample_prediction_th = sample_prediction_dann_th
            count_dann_ideal += 1

        #print("SAE: " + str(target_best_fm_cnn))
        #print("DANN: " + str(target_best_fm_dann))

        predicts_auto_ideal.append(best_sample_prediction_th)

    print (config.db1 + "->" + config.db2 )
    print ("Filters in target:")
    print("SAE: " + str(count_cnn))
    print("DANN: " + str(count_dann))
    print("Total samples: " + str(count_cnn + count_dann))
    print ("Ideal filters:")
    print("SAE: " + str(count_cnn_ideal))
    print("DANN: " + str(count_dann_ideal))
    print("Total samples: " + str(count_cnn_ideal + count_dann_ideal))

    predicts_auto = np.asarray(predicts_auto)
    predicts_auto_ideal = np.asarray(predicts_auto_ideal)
    assert(len(predicts_auto)==len(predicts_auto_ideal))
    return predicts_auto, predicts_auto_ideal
        
def predictAUTODann(
                model_dann, 
                model_cnn, 
                config, 
                array_files_to_save, 
                threshold_cnn, 
                threshold_dann, 
                correlation_type, threshold_correl, 
                histogram_source_cnn, 
                histogram_target_cnn,
                num_decimal=None):
    print('Calculating AUTODANN...')

    #Histogram for source
    list_histogram_source_cnn = [histogram_source_cnn[round(float(number)/len(histogram_source_cnn), 1)] for number in range(len(histogram_source_cnn))]
    number_pixels_source = sum(list_histogram_source_cnn)
    normalized_list_histogram_source_cnn = [number / float(number_pixels_source) for number in list_histogram_source_cnn]

    list_histogram_target_cnn = [histogram_target_cnn[round(float(number)/len(histogram_target_cnn), 1)] for number in range(len(histogram_target_cnn))]
    number_pixels_target = sum(list_histogram_target_cnn)
    normalized_list_histogram_target_cnn = [number / float(number_pixels_target) for number in list_histogram_target_cnn]

    print ("------------------------Normalized global histograms---------------------------")
    print (config.db1)
    print(normalized_list_histogram_source_cnn)
    print (config.db2)
    print(normalized_list_histogram_target_cnn)


    array_files = load_array_of_files(config.path, array_files_to_save)

    list_target_best_fm_autodann = []
    list_target_results_autodann = []
    list_target_best_fm_cnn = []
    list_target_best_fm_dann = []
    list_target_best_fm_ideal = []

    count_cnn = 0
    count_dann = 0
    count_cnn_ideal = 0
    count_dann_ideal = 0

    for fname in array_files:
        print('Processing image', fname)

        fname_gt = fname.replace(utilConst.X_SUFIX, utilConst.Y_SUFIX)
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(fname_gt, cv2.IMREAD_GRAYSCALE)

        if img.shape[0] < config.window or img.shape[1] < config.window:
            new_rows = config.window if img.shape[0] < config.window else img.shape[0]
            new_cols = config.window if img.shape[1] < config.window else img.shape[1]
            img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

        #cv2.imshow("img", img)
        #cv2.waitKey(0)

        finalImg = np.zeros(img.shape, dtype=float)
        finalImg_bin = np.zeros(img.shape, dtype=float)
        finalImg_sel = np.zeros(img.shape, dtype=float)
        finalImg_ideal = np.zeros(img.shape, dtype=float)

        finalImg_cnn = np.zeros(img.shape, dtype=float)
        finalImg_dann = np.zeros(img.shape, dtype=float)
        
        finalImg_bin_cnn = np.zeros(img.shape, dtype=float)
        finalImg_bin_dann = np.zeros(img.shape, dtype=float)
        finalImg_bin_ideal = np.zeros(img.shape, dtype=float)
        
        for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=config.window-5, windowSize=(config.window, config.window)):
            if window.shape[0] != config.window or window.shape[1] != config.window:
                continue

            roi = img[y:(y + config.window), x:(x + config.window)].copy()

            #cv2.imshow("roi", roi)
            #cv2.waitKey(0)

            roi = roi.reshape(1, config.window, config.window, 1)
            roi = roi.astype('float32')
            norm_type = '255'
            roi = utilDataGenerator.normalize_data( roi, norm_type )

            prediction_cnn = model_cnn.predict(roi)
            prediction_cnn = prediction_cnn[:,2:prediction_cnn.shape[1]-2,2:prediction_cnn.shape[2]-2,:]
            sample_prediction_cnn = prediction_cnn[0].reshape(config.window-4, config.window-4)

            #Histogram for target
            histogram_prediction_cnn = getHistogramBins(prediction_cnn, num_decimal)
            list_histogram_prediction_cnn = histogram_prediction_cnn.values()
            number_pixels_target = sum(list_histogram_prediction_cnn)

            normalized_list_histogram_prediction_cnn = [number / float(number_pixels_target) for number in list_histogram_prediction_cnn]
            
            correlation = get_correlation_metric(correlation_type, normalized_list_histogram_prediction_cnn, normalized_list_histogram_source_cnn)
            #get_all_correlation_metrics(normalized_list_histogram_prediction_cnn, normalized_list_histogram_source_cnn)

            prediction_dann = model_dann.predict(roi)
            prediction_dann = prediction_dann[:,2:prediction_dann.shape[1]-2,2:prediction_dann.shape[2]-2,:]
            sample_prediction_dann = prediction_dann[0].reshape(config.window-4, config.window-4)
            
            if correlation > threshold_correl:
                #SAE
                threshold = threshold_cnn
                sample_prediction = sample_prediction_cnn
                count_cnn += 1

            else:
                #DANN
                threshold = threshold_dann
                sample_prediction = sample_prediction_dann
                finalImg_sel[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction >= 0.0)
                count_dann += 1
            
            finalImg[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction
            finalImg_bin[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction < threshold)

            finalImg_cnn[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction_cnn
            finalImg_bin_cnn[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction_cnn < threshold_cnn)

            finalImg_dann[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction_dann
            finalImg_bin_dann[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction_dann < threshold_dann)
            
           
            gt_sample = gt[y+2:(y + config.window-2), x+2:(x + config.window-2)]
            sample_prediction_cnn_th = sample_prediction_cnn < threshold_cnn
            sample_prediction_dann_th = sample_prediction_dann < threshold_dann

            target_best_fm_cnn, _, target_precision_cnn, target_recall_cnn = utilMetrics.calculate_best_fm(sample_prediction_cnn_th, gt_sample > 0.5, None, False)
            target_best_fm_dann, _, target_precision_dann, target_recall_dann = utilMetrics.calculate_best_fm(sample_prediction_dann_th, gt_sample > 0.5, None, False)

            if target_best_fm_cnn > target_best_fm_dann:
                best_sample_prediction_th = sample_prediction_cnn_th
                best_sample_prediction = sample_prediction_cnn
                count_cnn_ideal += 1
                best_threshold_auto_ideal = threshold_cnn
            else:
                best_sample_prediction_th = sample_prediction_dann_th
                best_sample_prediction = sample_prediction_dann
                count_dann_ideal += 1
                best_threshold_auto_ideal = threshold_dann

            finalImg_bin_ideal[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (best_sample_prediction_th)

            #cv2.imshow("finalImg", (1 - finalImg.astype('uint8')) * 255 )
            #cv2.waitKey(0)
        
        import ntpath
        filename = ntpath.basename(fname)
        filename_out = filename.replace(".", "_"+str(config.type) + ".")
        filename_out_sel = filename.replace(".", "_"+str(config.type) + "_sel.")
        filename_out_cnn = filename.replace(".", "_"+str(config.type) + "_cnn.")
        filename_out_dann = filename.replace(".", "_"+str(config.type) + "_dann.")
        

        pathdir_outimage = "OUTPUT/auto_dann/probs/" + str(config.db1) +"-"+ str(config.db2) + "/"

        util.mkdirp( os.path.dirname(pathdir_outimage) )
        cv2.imwrite(pathdir_outimage + str(filename_out), finalImg_bin*255)
        cv2.imwrite(pathdir_outimage + str(filename), 255-img*255)
        cv2.imwrite(pathdir_outimage + str(filename_out_sel), finalImg_sel*255)

        cv2.imwrite(pathdir_outimage + str(filename_out_cnn), finalImg_bin_cnn*255)
        cv2.imwrite(pathdir_outimage + str(filename_out_dann), finalImg_bin_dann*255)
        
        histogram_prediction = getHistogramBins(finalImg, num_decimal)

        out_histogram_filename = fname.replace(config.path, 'OUTPUT/auto_dann/histogram')

        out_histogram_filename = str(out_histogram_filename)
        out_histogram_filename = out_histogram_filename.replace(utilConst.X_SUFIX+'/', '/'+config.modelpath + '/')
        out_histogram_filename = out_histogram_filename.replace(utilConst.WEIGHTS_DANN_FOLDERNAME+'/', '')
        out_histogram_filename = out_histogram_filename.replace(utilConst.WEIGHTS_CNN_FOLDERNAME+'/', '')
        out_histogram_filename = out_histogram_filename.replace(utilConst.LOGS_DANN_FOLDERNAME+'/', '')
        out_histogram_filename = out_histogram_filename.replace(utilConst.LOGS_CNN_FOLDERNAME+'/', '')

        out_histogram_filename = out_histogram_filename.replace('.h5', '/OUT_PR').replace('.npy', '/OUT_PR')

        out_histogram_filename = out_histogram_filename.replace(".png", ".txt")
        print(' - Saving predicted image to:', out_histogram_filename)
        util.mkdirp( os.path.dirname(out_histogram_filename) )

        items_histogram = sorted(histogram_prediction.items())
        str_prob = ""
        str_value = ""

        for prob, value in items_histogram:
            str_prob += str(prob) + "\t"
            str_value += str(value-1) + "\t"

        str_histogram = str_prob + "\n" + str_value

        saveString(str_histogram, out_histogram_filename, True)

        finalImg_bin = (finalImg_bin<0.5)
        finalImg_bin_cnn = (finalImg_bin_cnn<0.5)
        finalImg_bin_dann = (finalImg_bin_dann<0.5)
        finalImg_bin_ideal = (finalImg_bin_ideal<0.5)
        gt = (gt < 0.5)

        print ("F1 at sample-level:")

        target_best_fm_autodann, _, target_precision_autodann, target_recall_autodann = utilMetrics.calculate_best_fm(finalImg_bin, gt, None)
        print ("SAE:")
        target_best_fm_cnn, _, target_precision_cnn, target_recall_cnn = utilMetrics.calculate_best_fm(finalImg_bin_cnn, gt, None)
        print ("DANN:")
        target_best_fm_dann, _, target_precision_dann, target_recall_dann = utilMetrics.calculate_best_fm(finalImg_bin_dann, gt, None)
        print ("IDEAL:")
        target_best_fm_ideal, _, target_precision_dann, target_recall_dann = utilMetrics.calculate_best_fm(finalImg_bin_ideal, gt, None)

        list_target_best_fm_autodann.append(target_best_fm_autodann)
        list_target_results_autodann.append(( target_best_fm_autodann, target_precision_autodann, target_recall_autodann ))

        list_target_best_fm_cnn.append(target_best_fm_cnn)
        list_target_results_cnn.append(( target_best_fm_cnn, target_precision_cnn, target_recall_cnn ))

        list_target_best_fm_dann.append(target_best_fm_dann)
        list_target_results_dann.append(( target_best_fm_dann, target_precision_dann, target_recall_dann ))

        list_target_best_fm_ideal.append(target_best_fm_ideal)
        list_target_results_ideal.append(( target_best_fm_ideal, target_precision_dann, target_recall_dann ))

    print ("F1 at page-level")
    
    str_f1 = "SAE:\t"
    str_f1 += str(list_target_results_cnn)
    
    str_f1 += ("\nDANN:\t")
    str_f1 += (str(list_target_results_dann))
    
    str_f1 +=  ("\nAUTODANN:\t")
    str_f1 += (str(list_target_results_autodann))

    str_f1 +=  ("\IDEAL:\t")
    str_f1 += (str(list_target_results_ideal))
    
    avg_cnn = np.average(list_target_best_fm_cnn)
    avg_dann = np.average(list_target_best_fm_dann)
    avg_autodann = np.average(list_target_best_fm_autodann)
    avg_autodann_ideal = np.average(list_target_best_fm_ideal)
    
    str_f1 +=  ("\n------------------------------------------------------------\n")
    str_f1 +=  ('PAGE-LEVEL F1 SAE\tDANN\tAutoDANN\tIDEAL\n')
    str_f1 +=  (str(avg_cnn) + "\t")
    str_f1 +=  (str(avg_dann) + "\t")
    str_f1 +=  (str(avg_autodann) + "\t")
    str_f1 +=  (str(avg_autodann_ideal))
    
    str_f1 = str_f1.replace(".", ",")

    print(str_f1)
    pathdir_F1 = "OUTPUT/auto_dann/probs/" + str(config.db1) +"-"+ str(config.db2) + "/" + "f1.txt"
    saveString(str_f1, pathdir_F1, True)
    
    print (config.db1 + "->" + config.db2 )
    print ("Filters in target:")
    print("SAE: " + str(count_cnn))
    print("DANN: " + str(count_dann))
    print("Total samples: " + str(count_cnn + count_dann))
    print ("Ideal filters:")
    print("SAE: " + str(count_cnn_ideal))
    print("DANN: " + str(count_dann_ideal))
    print("Total samples: " + str(count_cnn_ideal + count_dann_ideal))

    return avg_cnn, avg_dann, avg_autodann


def saveString(content_string, path_file, close_file):
    assert type(content_string) == str
    assert type(path_file) == str
    assert type(close_file) == bool
    
    path_dir = os.path.dirname(path_file)

    if (path_dir != ""):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, 493)
            
    f = open(path_file,"w+")
    f.write(content_string)
    
    if (close_file == True):
        f.close()
        
def readString(path_file):
    assert type(path_file) == str
    
    path_dir = os.path.dirname(path_file)

    if (path_dir != ""):
        if not os.path.exists(path_dir):
            return False
    try:  
        f = open(path_file,"r")
    except:
        return False
    str_lines = f.readlines()
    
    f.close()
    
    return str_lines

# ----------------------------------------------------------------------------
def save_images(model, array_files_to_save, config):
    assert config.threshold != -1

    PONDERATE = True

    print('Saving images...')

    array_files = load_array_of_files(config.path, array_files_to_save)

    for fname in array_files:
        print('Processing image', fname)

        fname_gt = fname.replace(utilConst.X_SUFIX, utilConst.Y_SUFIX)
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(fname_gt, cv2.IMREAD_GRAYSCALE)

        rows = img.shape[0]
        cols = img.shape[1]
        if img.shape[0] < config.window or img.shape[1] < config.window:
            new_rows = config.window if img.shape[0] < config.window else img.shape[0]
            new_cols = config.window if img.shape[1] < config.window else img.shape[1]
            img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

        #cv2.imshow("img", img)
        #cv2.waitKey(0)

        if PONDERATE == False:
            finalImg = np.zeros(img.shape, dtype=bool)
        else:
            finalImg = np.zeros(img.shape, dtype=float)
        finalWeights = np.zeros(img.shape, dtype=float)

        for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=config.step-11, windowSize=(config.window, config.window)):
            if window.shape[0] != config.window or window.shape[1] != config.window:
                continue

            roi = img[y:(y + config.window), x:(x + config.window)].copy()

            #cv2.imshow("roi", roi)
            #cv2.waitKey(0)

            roi = roi.reshape(1, config.window, config.window, 1)
            roi = roi.astype('float32')
            norm_type = '255'
            roi = utilDataGenerator.normalize_data( roi, norm_type )

            prediction = model.predict(roi)
            prediction = prediction[:,5:prediction.shape[1]-5,5:prediction.shape[2]-5,:]

            if PONDERATE == False:  #SIN PONDERACIÓN
                prediction = (prediction > config.threshold)
                finalImg[y+5:(y + config.window-5), x+5:(x + config.window-5)] = prediction[0].reshape(config.window-10, config.window-10)
            else:
                # CON PONDERACIÓN
                finalImg[y+5:(y + config.window-5), x+5:(x + config.window-5)] += prediction[0].reshape(config.window-10, config.window-10)
                finalWeights[y+5:(y + config.window-5), x+5:(x + config.window-5)] += 1

            #cv2.imshow("finalImg", (1 - finalImg.astype('uint8')) * 255 )
            #cv2.waitKey(0)

        if PONDERATE == True:
            finalImg /=  finalWeights.astype('float32')
            finalImg = (finalImg > config.threshold)

        finalImg = 1 - finalImg.astype('uint8')
        finalImg *= 255
        #finalImg = finalImg.astype('uint8')

        if finalImg.shape[0] != rows or finalImg.shape[1] != cols:
            finalImg = cv2.resize(finalImg, (cols, rows), interpolation = cv2.INTER_CUBIC)

        img_diff = __calculate_img_diff(finalImg, gt)

        # Save image...
        outFilename = fname.replace(config.path, 'OUTPUT')
        outFilename = outFilename.replace(utilConst.X_SUFIX+'/', '/'+config.modelpath + '/')
        outFilename = outFilename.replace(utilConst.WEIGHTS_DANN_FOLDERNAME+'/', '')
        outFilename = outFilename.replace(utilConst.WEIGHTS_CNN_FOLDERNAME+'/', '')
        outFilename = outFilename.replace(utilConst.LOGS_DANN_FOLDERNAME+'/', '')
        outFilename = outFilename.replace(utilConst.LOGS_CNN_FOLDERNAME+'/', '')

        outFilenamePred = outFilename.replace('.h5', '/OUT_PR').replace('.npy', '/OUT_PR')
        outFilenameDiff = outFilename.replace('.h5', '/OUT_DIFF').replace('.npy', '/OUT_DIFF')

        print(' - Saving predicted image to:', outFilenamePred)
        print(' - Saving Diff image to:', outFilenameDiff)
        util.mkdirp( os.path.dirname(outFilenamePred) )
        util.mkdirp( os.path.dirname(outFilenameDiff) )
        cv2.imwrite(outFilenamePred, finalImg)
        cv2.imwrite(outFilenameDiff, img_diff)
