# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings

gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']=gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
#warnings.filterwarnings('ignore')

import copy
import os
import argparse
import numpy as np
import util
import utilConst
import utilIO
import utilMetrics
import utilDataGenerator
import utilCNN
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


util.mkdirp( utilConst.WEIGHTS_CNN_FOLDERNAME + '/truncated')
util.mkdirp( utilConst.WEIGHTS_DANN_FOLDERNAME + '/truncated')


# ----------------------------------------------------------------------------
def menu():
    parser = argparse.ArgumentParser(description='DA SAE')
    parser.add_argument('-type',   default='dann', type=str,     choices=['dann', 'cnn', 'autodann'],  help='Training type')

    parser.add_argument('-path',  required=True,   help='base path to datasets')
    parser.add_argument('-db1',       required=True,  choices=utilConst.ARRAY_DBS, help='Database name')
    parser.add_argument('-db2',       required=True,  choices=utilConst.ARRAY_DBS, help='Database name')

    parser.add_argument('--aug',   action='store_true', help='Load augmentation folders')
    parser.add_argument('-w',          default=256,    dest='window',           type=int,   help='window size')
    parser.add_argument('-s',          default=-1,      dest='step',                type=int,   help='step size. -1 to use window size')

    parser.add_argument('-gpos',          default=0,        dest='grl_position',     type=int,   help='Position of GRL')

    parser.add_argument('-l',          default=4,        dest='nb_layers',     type=int,   help='Number of layers')
    parser.add_argument('-f',          default=64,      dest='nb_filters',   type=int,   help='nb_filters')
    parser.add_argument('-k',          default=5,        dest='k_size',            type=int,   help='kernel size')
    parser.add_argument('-drop',   default=0,        dest='dropout',          type=float, help='dropout value')

    parser.add_argument('-lda',      default=0.001,    type=float,    help='Reversal gradient lambda')
    parser.add_argument('-lda_inc',  default=0.001,    type=float,    help='Reversal gradient lambda increment per epoch')
    parser.add_argument('-page',   default=-1,      type=int,   help='Nb pages to divide the training set. -1 to load all')
    parser.add_argument('-super',  default=1,      dest='nb_super_epoch',      type=int,   help='nb_super_epoch')
    parser.add_argument('-th',         default=-1,     dest='threshold',           type=float, help='threshold. -1 to test from 0 to 1')
    parser.add_argument('-e',           default=200,    dest='epochs',            type=int,   help='nb_epoch')
    parser.add_argument('-b',           default=10,     dest='batch',               type=int,   help='batch size')
    parser.add_argument('-verbose',     default=1,                                  type=int,   help='1=show batch increment, other=mute')

    parser.add_argument('--truncate',   action='store_true', help='Truncate data')
    parser.add_argument('--test',   action='store_true', help='Only run test')
    parser.add_argument('--show',   action='store_true', help='Show the result')
    parser.add_argument('--tboard',   action='store_true', help='Active tensorboard')
    parser.add_argument('--save',   action='store_true', help='Save binarized output images')
    parser.add_argument('-loadmodel', type=str,   help='Weights filename to load for test')

    parser.add_argument('-d_model',    default=0,    dest='domain_model_version', type=int,   help='Domain model version')

    parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')

    args = parser.parse_args()

    if args.step == -1:
        args.step = args.window

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args


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
    x_test, y_test = utilDataGenerator.generate_chunks(array_test_files, window, window)

    train_data_generator = None
    if is_test == False:
        array_train_files = utilIO.load_array_of_files(path, train_folds, truncate)
        train_data_generator = utilDataGenerator.LazyChunkGenerator(array_train_files, page_size, window, step)
        train_data_generator.shuffle()

    print('DB: ', db)
    print(' - Train data files:', len(array_train_files) if is_test == False else '--')
    print(' - Test data files:', len(array_test_files))
    print(' - Test data shape:', x_test.shape)

    return {'name': db,
                     'generator': train_data_generator,
                     'x_test': x_test, 'y_test': y_test}


# -----------------------------------------------------------------------------
def train_and_evaluate(datasets, input_shape, config):
    summary = config.type == 'dann'
    weights_filename = None
    model_dann = utilDANNModel.DANNModel(input_shape, config, summary)
    model_cnn = utilDANNModel.DANNModel(input_shape, config, summary)

    if config.type == 'dann':
        dann = model_dann
        weights_filename = utilDANN.get_dann_weights_filename( utilConst.WEIGHTS_DANN_FOLDERNAME,
                                                                                                        datasets['source']['name'],
                                                                                                        datasets['target']['name'], config)
        if config.test == False:
            print('Train SAE DANN...')
            utilDANN.train_dann(dann, datasets['source'], datasets['target'],
                                                        weights_filename,
                                                        utilConst.LOGS_DANN_FOLDERNAME,
                                                        utilConst.CSV_LOGS_DANN_FOLDERNAME,
                                                        config)
        else:
            #weights_filename = weights_filename.replace("_dmodel2", "")
            print(weights_filename)
            dann.load( weights_filename )  # Load the last save weights...

    elif config.type == 'cnn':
        dann = model_cnn
        print('Train SAE (without DA)...')
        print(dann.label_model.summary())
        weights_filename = utilCNN.get_cnn_weights_filename( utilConst.WEIGHTS_CNN_FOLDERNAME, datasets['source']['name'], config)

        if config.test == False:
            utilCNN.train_cnn(dann.label_model,  datasets['source'], datasets['target'],
                                                    weights_filename,
                                                    utilConst.LOGS_CNN_FOLDERNAME,
                                                    utilConst.CSV_LOGS_CNN_FOLDERNAME,
                                                    config)
        else:
            
            dann.label_model.load_weights(weights_filename)
    elif config.type == 'autodann':
        assert(config.test == True)
        dann = None

        config_dann = copy.deepcopy(config)
        config_dann.type = 'dann'

        config_cnn = copy.deepcopy(config)
        config_cnn.type = 'cnn'

        weights_filename_dann = utilDANN.get_dann_weights_filename( utilConst.WEIGHTS_DANN_FOLDERNAME, datasets['source']['name'], datasets['target']['name'], config_dann)
        weights_filename_cnn = utilCNN.get_cnn_weights_filename( utilConst.WEIGHTS_CNN_FOLDERNAME, datasets['source']['name'], config_cnn)

        model_dann.load( weights_filename_dann )
        model_cnn.label_model.load_weights(weights_filename_cnn)

    else:
        raise Exception('Unknown type')

    batch=1
    if dann is not None:
        print('# Evaluate...')
        source_loss, source_mse = dann.label_model.evaluate(datasets['source']['x_test'], datasets['source']['y_test'], batch_size=batch, verbose=0)
        target_loss, target_mse = dann.label_model.evaluate(datasets['target']['x_test'], datasets['target']['y_test'], batch_size=batch, verbose=0)
        print('Result: {}\t{}\t{:.4f}\t{:.4f}'.format(datasets['source']['name'], datasets['target']['name'], source_mse, target_mse))

        pred_source = dann.label_model.predict(datasets['source']['x_test'], batch_size=batch, verbose=0)
        pred_target = dann.label_model.predict(datasets['target']['x_test'], batch_size=batch, verbose=0)
        print('SOURCE:')
        source_best_fm, source_best_th = utilMetrics.calculate_best_fm(pred_source, datasets['source']['y_test'])
        print('TARGET:')
        target_best_fm, target_best_th = utilMetrics.calculate_best_fm(pred_target, datasets['target']['y_test'], source_best_th)

        config.modelpath = weights_filename
        _, target_test_folds = utilIO.load_folds_names(config.db2)
        utilIO.getHistograms(dann.label_model, target_test_folds, config, source_best_th, 1)

        # Save output images
        if config.save:
            config.modelpath = weights_filename
            config.threshold = target_best_th
            _, target_test_folds = utilIO.load_folds_names(config.db2)
            utilIO.save_images(dann.label_model, target_test_folds, config)
    else:
        #AUTODANN

        pathdir_results = "OUTPUT/auto_dann/probs/" + str(config.db1) +"-"+ str(config.db2) + "/" + "results_sample_level.txt"

        content_results_file = utilIO.readString(pathdir_results)
        if content_results_file == False:

            pred_source_dann = model_dann.label_model.predict(datasets['source']['x_test'], batch_size=batch, verbose=0)
            pred_target_dann = model_dann.label_model.predict(datasets['target']['x_test'], batch_size=batch, verbose=0)

            pred_source_cnn = model_cnn.label_model.predict(datasets['source']['x_test'], batch_size=batch, verbose=0)
            pred_target_cnn = model_cnn.label_model.predict(datasets['target']['x_test'], batch_size=batch, verbose=0)

            print('SOURCE CNN:')
            source_best_fm_cnn, source_best_th_cnn = utilMetrics.calculate_best_fm(pred_source_cnn, datasets['source']['y_test'])
            print('TARGET CNN:')
            target_best_fm_cnn, target_best_th_cnn = utilMetrics.calculate_best_fm(pred_target_cnn, datasets['target']['y_test'], source_best_th_cnn)

            print('SOURCE DANN:')
            source_best_fm_dann, source_best_th_dann = utilMetrics.calculate_best_fm(pred_source_dann, datasets['source']['y_test'])
            print('TARGET DANN:')
            target_best_fm_dann, target_best_th_dann = utilMetrics.calculate_best_fm(pred_target_dann, datasets['target']['y_test'], source_best_th_dann)
        
            str_results = str(source_best_fm_cnn) + "\n"
            str_results += str(source_best_th_cnn) + "\n"
            
            str_results += str(target_best_fm_cnn) + "\n"
            str_results += str(target_best_th_cnn) + "\n"

            str_results += str(source_best_fm_dann) + "\n"
            str_results += str(source_best_th_dann) + "\n"
            
            str_results += str(target_best_fm_dann) + "\n"
            str_results += str(target_best_th_dann) + "\n"

            utilIO.saveString(str_results, pathdir_results, True)
        else:
            print('SOURCE CNN:')
            source_best_fm_cnn = float(content_results_file[0])
            source_best_th_cnn = float(content_results_file[1])
            print ("f1: " + str(source_best_fm_cnn) + "\tth: " + str(source_best_th_cnn))

            print('TARGET CNN:')
            target_best_fm_cnn = float(content_results_file[2])
            target_best_th_cnn = float(content_results_file[3])
            print ("f1: " + str(target_best_fm_cnn) + "\tth: " + str(target_best_th_cnn))

            print('SOURCE DANN:')
            source_best_fm_dann = float(content_results_file[4])
            source_best_th_dann = float(content_results_file[5])
            print ("f1: " + str(source_best_fm_dann) + "\tth: " + str(source_best_th_dann))
        
            print('TARGET DANN:')
            target_best_fm_dann = float(content_results_file[6])
            target_best_th_dann = float(content_results_file[7])
            print ("f1: " + str(target_best_fm_dann) + "\tth: " + str(target_best_th_dann))

        _, source_test_folds = utilIO.load_folds_names(config.db1)
        histogram_source_cnn = utilIO.getHistogramDomain(source_test_folds, model_cnn.label_model, config, 1)

        _, target_test_folds = utilIO.load_folds_names(config.db2)
        histogram_target_cnn = utilIO.getHistogramDomain(target_test_folds, model_cnn.label_model, config, 1)

        config.modelpath = "auto_dann_" + weights_filename_cnn + weights_filename_dann

        content_results_file = utilIO.readString(pathdir_results)
        target_best_fm_cnn = float(content_results_file[2])
        target_best_th_cnn = float(content_results_file[3])
        target_best_fm_dann = float(content_results_file[6])
        target_best_th_dann = float(content_results_file[7])
        print("Thresholds...SAE\tDANN")
        print("th: " + str(target_best_th_cnn) + "\t" + str(target_best_th_dann))

        #utilIO.predictModelAtFullPage(
        #                model_cnn.label_model, 
        #                config, 
        #                "sae",
        #                target_test_folds, 
        #                source_best_th_cnn, 
        #                1)

        #utilIO.predictModelAtFullPage(
        #                model_dann.label_model, 
        #                config, 
        #                "dann",
        #                target_test_folds, 
        #                source_best_th_dann, 
        #                1)

        pred_target_auto = utilIO.predictAutoDANN_AtSampleLevel(
                        model_dann.label_model, 
                        model_cnn.label_model,
                        config, 
                        datasets['target']['x_test'],
                        source_best_th_cnn, 
                        source_best_th_dann,
                        0.25, #threshold_correl_pearson, 
                        histogram_source_cnn,
                        1)

        pred_target_auto = pred_target_auto.reshape(datasets['target']['y_test'].shape)
        

        utilIO.predictAUTODann(
                        model_dann.label_model, 
                        model_cnn.label_model, 
                        config, 
                        target_test_folds, 
                        source_best_th_cnn, 
                        source_best_th_dann, 
                        0.25, #threshold_correl_pearson, 
                        histogram_source_cnn, 
                        histogram_target_cnn,
                        1)
        
        source_best_fm_auto, _ = utilMetrics.calculate_best_fm(pred_target_auto, datasets['target']['y_test'] > 0.5, None)
        
        print("------------------------------------------------------------")
        print('SAMPLE-LEVEL F1 SAE\tDANN\tAutoDANN:')
        print (str(target_best_fm_cnn).replace(".", ",") + "\t" + str(target_best_fm_dann).replace(".", ",") + "\t" + str(source_best_fm_auto).replace(".", ","))

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    config = menu()

    source_data = load_data(config.path, config.db1, config.window, config.step,
                                                        config.page, config.test, config.truncate)
    target_data = load_data(config.path, config.db2, config.window, config.step,
                                                        config.page, config.test, config.truncate)
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

    train_and_evaluate(datasets, input_shape, config)




