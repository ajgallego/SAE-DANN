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


# ----------------------------------------------------------------------------
def getHistograms(model, array_files_to_save, config, threshold = 0.5, num_decimal=None):

    print('Calculating histogram...')

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
        

        tuple_prediction = tuple(finalImg.reshape(1,-1)[0])

        if num_decimal is not None:
            tuple_prediction_round = []
            for num in tuple_prediction:
                if num > 0.01:
                    tuple_prediction_round.append(round(num, num_decimal))
                
            #tuple_prediction_round = [round(num, num_decimal) for num in tuple_prediction]
            tuple_prediction = tuple_prediction_round

            precision = 1.
            for i in range(num_decimal):
                precision /= 10.

            value = 0.
            value = round(value, num_decimal)
            while value <= 1:
                tuple_prediction.append(value)
                value += precision
                value = round(value, num_decimal)
        
        histogram_prediction = Counter(tuple_prediction)

        out_histogram_filename = fname.replace(config.path, 'OUTPUT/histogram')

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

        tuple_prediction_round = [str_prob for prob, value in items_histogram]

        saveString(str_histogram, out_histogram_filename, True)
        

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
           
        histogram_domain_fname = getHistogramBins(finalImg, num_decimal)
        print(str(histogram_domain_fname))

        if histogram_domain is None:
            histogram_domain = histogram_domain_fname.copy()
        else:
            histogram_domain = histogram_domain + histogram_domain_fname

        print(str(histogram_domain))

    return histogram_domain

def predictAUTODann(
                model_dann, 
                model_cnn, 
                config, 
                array_files_to_save, 
                target_data,
                threshold_cnn, 
                threshold_dann, 
                threshold_correl_pearson, 
                histogram_source_cnn, 
                num_decimal=None):
    print('Calculating AUTODANN...')

    #Histogram for source
    list_histogram_source_cnn = histogram_source_cnn.values()
    number_pixels_source = sum(list_histogram_source_cnn)
    normalized_list_histogram_source_cnn = [number / float(number_pixels_source) for number in list_histogram_source_cnn]

    array_files = load_array_of_files(config.path, array_files_to_save)

    list_target_best_fm_autodann = []
    list_target_best_fm_cnn = []
    list_target_best_fm_dann = []

    list_target_best_fm_autodann_inv = []
    list_target_best_fm_cnn_inv = []
    list_target_best_fm_dann_inv = []

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
        finalImg_bin = np.zeros(img.shape, dtype=float)
        finalImg_sel = np.zeros(img.shape, dtype=float)

        finalImg_cnn = np.zeros(img.shape, dtype=float)
        finalImg_dann = np.zeros(img.shape, dtype=float)
        finalImg_bin_cnn = np.zeros(img.shape, dtype=float)
        finalImg_bin_dann = np.zeros(img.shape, dtype=float)
        
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
            
            correl_pearson = np.corrcoef(normalized_list_histogram_prediction_cnn, normalized_list_histogram_source_cnn)[0, 1]
            
            prediction_dann = model_dann.predict(roi)
            prediction_dann = prediction_dann[:,2:prediction_dann.shape[1]-2,2:prediction_dann.shape[2]-2,:]
            sample_prediction_dann = prediction_dann[0].reshape(config.window-4, config.window-4)
            
            if correl_pearson > threshold_correl_pearson:
                #SAE
                threshold = threshold_cnn
                sample_prediction = sample_prediction_cnn

            else:
                #DANN
                threshold = threshold_dann
                sample_prediction = sample_prediction_dann
                finalImg_sel[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction >= 0.0)
            
            
            finalImg[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction
            finalImg_bin[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction < threshold)

            finalImg_cnn[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction_cnn
            finalImg_bin_cnn[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction_cnn < threshold_cnn)

            finalImg_dann[y+2:(y + config.window-2), x+2:(x + config.window-2)] = sample_prediction_dann
            finalImg_bin_dann[y+2:(y + config.window-2), x+2:(x + config.window-2)] = (sample_prediction_dann < threshold_dann)
            
           
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

        finalImg_bin = (finalImg_bin>0.5)
        finalImg_bin_cnn = (finalImg_bin_cnn>0.5)
        finalImg_bin_dann = (finalImg_bin_dann>0.5)
        gt = (gt > 0.5)

        finalImg_bin_inv = (finalImg_bin<=0.5)
        finalImg_bin_cnn_inv = (finalImg_bin_cnn<=0.5)
        finalImg_bin_dann_inv = (finalImg_bin_dann<=0.5)
        gt_inv = (gt <= 0.5)

        target_best_fm_autodann, _ = utilMetrics.calculate_best_fm(finalImg_bin, gt, None)
        print ("SAE:")
        target_best_fm_cnn, _ = utilMetrics.calculate_best_fm(finalImg_bin_cnn, gt, None)
        print ("DANN:")
        target_best_fm_dann, _ = utilMetrics.calculate_best_fm(finalImg_bin_dann, gt, None)

        target_best_fm_autodann_inv, _ = utilMetrics.calculate_best_fm(finalImg_bin_inv, gt_inv, None)
        print ("SAE:")
        target_best_fm_cnn_inv, _ = utilMetrics.calculate_best_fm(finalImg_bin_cnn_inv, gt_inv, None)
        print ("DANN:")
        target_best_fm_dann_inv, _ = utilMetrics.calculate_best_fm(finalImg_bin_dann_inv, gt_inv, None)

        list_target_best_fm_autodann.append(target_best_fm_autodann)
        list_target_best_fm_cnn.append(target_best_fm_cnn)
        list_target_best_fm_dann.append(target_best_fm_dann)

        list_target_best_fm_autodann_inv.append(target_best_fm_autodann_inv)
        list_target_best_fm_cnn_inv.append(target_best_fm_cnn_inv)
        list_target_best_fm_dann_inv.append(target_best_fm_dann_inv)

    print ("F1 at page-level")
    
    str_f1 = "SAE:\t"
    str_f1 += str(list_target_best_fm_cnn)
    
    str_f1 += ("\nDANN:\t")
    str_f1 += (str(list_target_best_fm_dann))
    
    str_f1 +=  ("\nAUTODANN:\t")
    str_f1 += (str(list_target_best_fm_autodann))

    str_f1 += "\n----------------------INVERTED-----------------------"
    str_f1 += "\nSAE:\t"
    str_f1 += str(list_target_best_fm_cnn_inv)
    
    str_f1 += ("\nDANN:\t")
    str_f1 += (str(list_target_best_fm_dann_inv))
    
    str_f1 +=  ("\nAUTODANN:\t")
    str_f1 += (str(list_target_best_fm_autodann_inv))


    avg_cnn = np.average(list_target_best_fm_cnn)
    avg_dann = np.average(list_target_best_fm_dann)
    avg_autodann = np.average(list_target_best_fm_autodann)
    
    avg_cnn_inv = np.average(list_target_best_fm_cnn_inv)
    avg_dann_inv = np.average(list_target_best_fm_dann_inv)
    avg_autodann_inv = np.average(list_target_best_fm_autodann_inv)

    str_f1 +=  ("\nAVERAGE- SAE - DANN - AUTODANN\n")
    str_f1 +=  (str(avg_cnn) + "\t")
    str_f1 +=  (str(avg_dann) + "\t")
    str_f1 +=  (str(avg_autodann))

    str_f1 +=  ("\nINVERTED AVERAGE- SAE - DANN - AUTODANN\n")
    str_f1 +=  (str(avg_cnn_inv) + "\t")
    str_f1 +=  (str(avg_dann_inv) + "\t")
    str_f1 +=  (str(avg_autodann_inv))

    print(str_f1)
    pathdir_F1 = "OUTPUT/auto_dann/probs/" + str(config.db1) +"-"+ str(config.db2) + "/" + "f1.txt"
    saveString(str_f1, pathdir_F1, True)
    
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
