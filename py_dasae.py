# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings

gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']=gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
warnings.filterwarnings('ignore')

import cv2
import os
import argparse
import numpy as np
import util
import utilIO
import utilMetrics
import utilDataGenerator
import utilDANN
import utilDANNModel
from keras import backend as K

util.init()

K.set_image_data_format('channels_last')

"""if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)"""

if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth=True
    #session_conf.intra_op_parallelism_threads = 1  # For reproducibility
    #session_conf.inter_op_parallelism_threads = 1  # For reproducibility
    sess = tf.Session(config=session_conf, graph=tf.get_default_graph())
    #sess = tf.Session(config=session_conf)
    K.set_session(sess)


x_sufix = '_GR'
y_sufix = '_GT'
WEIGHTS_CNN_FOLDERNAME = 'WEIGHTS_CNN'
WEIGHTS_DANN_FOLDERNAME = 'WEIGHTS_DANN'

util.mkdirp( WEIGHTS_CNN_FOLDERNAME + '/truncated')
util.mkdirp( WEIGHTS_DANN_FOLDERNAME + '/truncated')


# ----------------------------------------------------------------------------
def menu():
    parser = argparse.ArgumentParser(description='DA SAE')
    #parser.add_argument('-type',   default='dann', type=str,     choices=['dann', 'cnn'],  help='Training type')

    parser.add_argument('-path',  required=True,   help='base path to datasets')
    parser.add_argument('-db1',       required=True,  choices=utilIO.ARRAY_DBS, help='Database name')
    parser.add_argument('-db2',       required=True,  choices=utilIO.ARRAY_DBS, help='Database name')

    parser.add_argument('--aug',   action='store_true', help='Load augmentation folders')
    parser.add_argument('-w',          default=256,    dest='window',           type=int,   help='window size')
    parser.add_argument('-s',          default=-1,      dest='step',                type=int,   help='step size. -1 to use window size')

    parser.add_argument('-l',          default=4,        dest='nb_layers',     type=int,   help='Number of layers')
    parser.add_argument('-f',          default=64,      dest='nb_filters',   type=int,   help='nb_filters')
    parser.add_argument('-k',          default=5,        dest='k_size',            type=int,   help='kernel size')
    parser.add_argument('-drop',   default=0,        dest='dropout',          type=float, help='dropout value')

    parser.add_argument('-lda',      default=0.001,    type=float,    help='Reversal gradient lambda')
    parser.add_argument('-page',   default=-1,      type=int,   help='Page size to divide the training set. -1 to load all')
    parser.add_argument('-super',  default=1,      dest='nb_super_epoch',      type=int,   help='nb_super_epoch')
    parser.add_argument('-th',         default=-1,     dest='threshold',           type=float, help='threshold. -1 to test from 0 to 1')
    parser.add_argument('-e',           default=200,    dest='epochs',            type=int,   help='nb_epoch')
    parser.add_argument('-b',           default=10,     dest='batch',               type=int,   help='batch size')
    parser.add_argument('-verbose',     default=1,                                  type=int,   help='1=show batch increment, other=mute')

    parser.add_argument('--truncate',   action='store_true', help='Truncate data')
    parser.add_argument('--test',   action='store_true', help='Only run test')
    parser.add_argument('--show',   action='store_true', help='Show the result')
    parser.add_argument('-loadmodel', type=str,   help='Weights filename to load for test')

    parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')

    args = parser.parse_args()

    if args.step == -1:
        args.step = args.window

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args


# ----------------------------------------------------------------------------
def save_images(model, config):
    assert config.threshold != -1

    _, test_folds = utilIO.load_folds_names(config.db2)

    array_files = utilIO.load_array_of_files(config.path, test_folds)

    for fname in array_files:
        print('Processing image', fname)

        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        img = np.asarray(img)

        rows = img.shape[0]
        cols = img.shape[1]
        if img.shape[0] < config.window or img.shape[1] < config.window:
            new_rows = config.window if img.shape[0] < config.window else img.shape[0]
            new_cols = config.window if img.shape[1] < config.window else img.shape[1]
            img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

        img = np.asarray(img).astype('float32')
        img = 255. - img

        finalImg = np.zeros(img.shape, dtype=bool)

        for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=config.step, windowSize=(config.window, config.window)):
            if window.shape[0] != config.window or window.shape[1] != config.window:
                continue

            roi = img[y:(y + config.window), x:(x + config.window)].copy()
            roi = roi.reshape(1, config.window, config.window, 1)
            roi = roi.astype('float32') #/ 255.

            prediction = autoencoder.predict(roi)
            prediction = (prediction > config.threshold)

            finalImg[y:(y + config.window), x:(x + config.window)] = prediction[0].reshape(config.window, config.window)

        finalImg = 1 - finalImg
        finalImg *= 255

        finalImg = finalImg.astype('uint8')

        if finalImg.shape[0] != rows or finalImg.shape[1] != cols:
            finalImg = cv2.resize(finalImg, (cols, rows), interpolation = cv2.INTER_CUBIC)

        outFilename = fname.replace('_GR/', '_PR-' + config.modelpath + '/')

        util.mkdirp( os.path.dirname(outFilename) )

        cv2.imwrite(outFilename, finalImg)


# ----------------------------------------------------------------------------
def load_data(path, db, window, step, page_size, is_test, truncate):
    print('Loading data...')
    train_folds, test_folds = utilIO.load_folds_names(db)

    # augmentation ?
    """if config.aug == True:       # Add the augmented folders
        assert False, "not implemented"
        for f in list(train_folds):
            train_folds.append( util.rreplace(f, "/", "/aug_", 1) )"""

    array_test_files = utilIO.load_array_of_files(path, test_folds, truncate)
    x_test, y_test = utilDataGenerator.generate_chunks(array_test_files, x_sufix, y_sufix, window, window)

    train_data_generator = None
    if is_test == False:
        array_train_files = utilIO.load_array_of_files(path, train_folds, truncate)
        train_data_generator = utilDataGenerator.LazyChunkGenerator(array_train_files, x_sufix, y_sufix, page_size, window, step)
        train_data_generator.shuffle()
        """if config.start_from > 0:
            train_data_generator.set_pos(config.start_from)"""

    print('DB: ', db)
    print(' - Train data files:', len(array_train_files) if is_test == False else '--')
    print(' - Test data files:', len(array_test_files))
    print(' - Test data shape:', x_test.shape)

    return {'name': db,
                     'generator': train_data_generator,
                     'x_test': x_test, 'y_test': y_test}


# -----------------------------------------------------------------------------
def run_dann(datasets, input_shape, weights_foldername, config):
    summary = True
    dann = utilDANNModel.DANNModel(input_shape, config, summary)

    weights_filename = utilDANN.get_dann_weights_filename( weights_foldername,
                                                                                                        datasets['source']['name'],
                                                                                                        datasets['target']['name'], config)

    if config.test == False:
        print('Train SAE DANN...')
        utilDANN.train_dann(dann, datasets['source'], datasets['target'],
                                                    config.page, config.nb_super_epoch,
                                                    config.epochs, config.batch, weights_filename,
                                                    config.lda)

    print('# Evaluate...')
    dann.load( weights_filename )  # Load the last save weights...
    source_loss, source_mse = dann.label_model.evaluate(datasets['source']['x_test'], datasets['source']['y_test'], batch_size=32, verbose=0)
    target_loss, target_mse = dann.label_model.evaluate(datasets['target']['x_test'], datasets['target']['y_test'], batch_size=32, verbose=0)
    print('Result: {}\t{}\t{:.4f}\t{:.4f}'.format(datasets['source']['name'], datasets['target']['name'], source_mse, target_mse))

    pred_source = dann.label_model.predict(datasets['source']['x_test'], batch_size=32, verbose=0)
    pred_target = dann.label_model.predict(datasets['target']['x_test'], batch_size=32, verbose=0)
    print('SOURCE:')
    utilMetrics.calculate_best_fm(pred_source, datasets['source']['y_test'])
    print('TARGET:')
    utilMetrics.calculate_best_fm(pred_target, datasets['target']['y_test'])


    # Save output images

    #config.modelpath = weights_filename
    #config.threshold = best_th
    save_images(dann.label_model, config)




# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    config = menu()

    source_data = load_data(config.path, config.db1, config.window, config.step, config.page, config.test, config.truncate)
    target_data = load_data(config.path, config.db2, config.window, config.step, config.page, config.test, config.truncate)
    datasets = {'source': source_data, 'target': target_data}

    print('SOURCE: {} \ttrain_generator:{}\tx_test:{}\ty_test:{}'.format(
        datasets['source']['name'],
        len(datasets['source']['generator']) if config.test == False else '--',
        datasets['source']['x_test'].shape,
        datasets['source']['y_test'].shape))
    print('TARGET: {} \ttrain_generator:{}\tx_test:{}\ty_test:{}'.format(
        datasets['target']['name'],
        len(datasets['target']['generator']) if config.test == False else '--',
        datasets['target']['x_test'].shape,
        datasets['target']['y_test'].shape))

    assert config.test == True or len(datasets['source']['generator']) > 0
    assert len(datasets['source']['x_test']) > 0
    assert len(datasets['source']['x_test']) == len(datasets['source']['y_test'])
    assert config.test == True or len(datasets['target']['generator']) > 0
    assert len(datasets['target']['x_test']) > 0
    assert len(datasets['target']['x_test']) == len(datasets['target']['y_test'])

    input_shape = datasets['source']['x_test'].shape[1:]
    assert input_shape == datasets['target']['x_test'].shape[1:]
    print(' - Input shape:', input_shape)


    run_dann(datasets, input_shape, WEIGHTS_DANN_FOLDERNAME, config)


