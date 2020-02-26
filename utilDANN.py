# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model
import tensorflow as tf
import util
import utilMetrics


# ----------------------------------------------------------------------------
def get_dann_weights_filename(folder, from_dataset, to_dataset, config):
    #### NEW MODEL ####
    #return '{}/weights_dann_model_from_{}_to_{}_e{}_b{}_lda{}.npy'.format(
    return '{}/weights_dannCONV_model_from_{}_to_{}_e{}_b{}_lda{}.npy'.format(
                            folder,
                            #('/truncated' if config.truncate else ''),
                            from_dataset, to_dataset,
                            str(config.epochs), str(config.batch), str(config.lda))

    """return "{}_{}_{}x{}_s{}{}{}_f{}_k{}{}_se{}_e{}_b{}_es{}".format(
                                config.db, config.dbp,
                                config.window, config.window, config.step,
                                '_aug' if config.aug else '',
                                '_drop'+str(config.dropout) if config.dropout > 0 else '',
                                config.nb_filters,
                                config.k_size,
                                '_s' + str(config.stride) if config.stride > 1 else '',
                                config.nb_super_epoch, config.nb_epoch, config.batch,
                                arconfiggs.early_stopping_mode)"""

# ----------------------------------------------------------------------------
def batch_generator(x_data, y_data=None, batch_size=1, shuffle_data=True):
    len_data = len(x_data)
    index_arr = np.arange(len_data)
    if shuffle_data:
        np.random.shuffle(index_arr)

    start = 0
    while len_data > start + batch_size:
        batch_ids = index_arr[start:start + batch_size]
        start += batch_size
        if y_data is not None:
            x_batch = x_data[batch_ids]
            y_batch = y_data[batch_ids]
            yield x_batch, y_batch
        else:
            x_batch = x_data[batch_ids]
            yield x_batch


# ----------------------------------------------------------------------------
def train_dann_batch(dann_model, src_generator, target_genenerator, target_x_train, batch_size):
    #### NEW MODEL ####
    #domain0 = np.zeros(target_x_train.shape[1:], dtype=int)
    #domain1 = np.ones(target_x_train.shape[1:], dtype=int)
    batchYd = np.concatenate(( np.zeros((batch_size // 2,) + target_x_train.shape[1:], dtype=int),
                                                              np.ones((batch_size // 2,) + target_x_train.shape[1:], dtype=int)) )
    #   np.tile(domain0, [batch_size // 2, 1, 1, 1]),
    #                                                       np.tile(domain1, [batch_size // 2, 1, 1, 1])))
    #print(batchYd)
    #print(batchYd.shape)

    for batchXs, batchYs in src_generator:
        try:
            batchXd = next(target_genenerator)
        except: # Restart...
            target_genenerator = batch_generator(target_x_train, None, batch_size=batch_size // 2)
            batchXd = next(target_genenerator)

        # Combine the labeled and unlabeled data along with the discriminative results
        combined_batchX = np.concatenate((batchXs, batchXd))
        batch2Ys = np.concatenate((batchYs, batchYs))

        #### NEW MODEL ####
        #batchYd = np.concatenate((np.tile([0, 1], [batch_size // 2, 1]),
        #                                                            np.tile([1, 0], [batch_size // 2, 1])))
        #print(combined_batchX.shape, batch2Ys.shape, batchYd.shape)

        result = dann_model.train_on_batch(combined_batchX,
                                                                                     {'classifier_output': batch2Ys,
                                                                                       'domain_output':batchYd})

    #print(dann_model.metrics_names)
    # print(result)
    return result


# ----------------------------------------------------------------------------
def __train_dann_page(dann_builder, source_x_train, source_y_train, source_x_test, source_y_test,
                                                 target_x_train, target_y_train, target_x_test, target_y_test,
                                                nb_epochs, batch_size, weights_filename):
    best_label_mse = np.inf
    target_genenerator = batch_generator(target_x_train, None, batch_size=batch_size // 2)

    for e in range(nb_epochs):
        src_generator = batch_generator(source_x_train, source_y_train, batch_size=batch_size // 2)

        # Update learning rates
        lr = float(K.get_value(dann_builder.opt.lr))* (1. / (1. + float(K.get_value(dann_builder.opt.decay)) * float(K.get_value(dann_builder.opt.iterations)) ))
        print(' - Lr:', lr, ' / Lambda:', dann_builder.grl_layer.get_hp_lambda())

        dann_builder.grl_layer.increment_hp_lambda_by(1e-7)      #1e-6  1e-4)  # !!!  ### NEW MODEL ####

        # Train batch
        loss, domain_loss, label_loss, domain_acc, label_mse = train_dann_batch(
                                            dann_builder.dann_model, src_generator, target_genenerator, target_x_train, batch_size )

        source_prediction = dann_builder.label_model.predict(source_x_train, batch_size=32, verbose=0)
        source_f1, source_th = utilMetrics.calculate_best_fm(source_prediction, source_y_train)

        saved = ""
        if label_mse <= best_label_mse:
            best_label_mse = label_mse
            dann_builder.save(weights_filename)
            saved = "SAVED"

        #target_loss, target_mse = dann_builder.label_model.evaluate(target_x_train, target_y_train, batch_size=32, verbose=0)
        target_loss, target_mse = dann_builder.label_model.evaluate(target_x_test, target_y_test, batch_size=32, verbose=0)

        y_pred = dann_builder.label_model.predict(target_x_test, batch_size=32, verbose=0)
        target_f1, target_th = utilMetrics.calculate_best_fm(y_pred, target_y_test)

        #print("Epoch [{}/{}]: source label loss={:.4f}, mse={:.4f} | domain loss={:.4f}, acc={:.4f} | target label loss={:.4f}, mse={:.4f} | {}".format(
        #                    e+1, nb_epochs, label_loss, label_mse, domain_loss, domain_acc, target_loss, target_mse, saved))
        print("Epoch [{}/{}]: source label loss={:.4f}, mse={:.4f}, f1={:.4f} | domain loss={:.4f}, acc={:.4f} | target label loss={:.4f}, mse={:.4f}, f1={:.4f} | {}".format(
                            e+1, nb_epochs, label_loss, label_mse, source_f1, domain_loss, domain_acc, target_loss, target_mse, target_f1, saved))


        gc.collect()



# ----------------------------------------------------------------------------
def train_dann(dann_builder, source, target, page, nb_super_epoch, nb_epochs,
                                    batch_size, weights_filename, initial_hp_lambda=0.01):
    print('Training DANN model...')

    dann_builder.grl_layer.set_hp_lambda(initial_hp_lambda)

    for se in range(nb_super_epoch):
        print(80 * "-")
        print("SUPER EPOCH: %03d/%03d" % (se+1, nb_super_epoch))

        source['generator'].reset()
        source['generator'].shuffle()

        for source_x_train, source_y_train in source['generator']:
            try:
                target_x_train, target_y_train = next(target['generator'])
            except: # Restart...
                target['generator'].reset()
                target['generator'].shuffle()
                target_x_train, target_y_train = next(target['generator'])

            print("> Train on %03d source page samples and %03d target page samples..." % (len(source_x_train), len(target_x_train)))
            #print(source_x_train.shape)
            #print(source_y_train.shape)
            #print(target_x_train.shape)

            __train_dann_page(dann_builder, source_x_train, source_y_train, source['x_test'], source['y_test'],
                                                    target_x_train, target_y_train, target['x_test'], target['y_test'],
                                                    nb_epochs, batch_size, weights_filename)

