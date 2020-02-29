# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import gc
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import utilMetrics
import util


# ----------------------------------------------------------------------------
def get_cnn_weights_filename(folder, dataset_name, config):
    return '{}{}/weights_cnn_model_{}_w{}_s{}_l{}_f{}_k{}_drop{}_page{}_e{}_b{}.npy'.format(
                            folder,
                            ('/truncated' if config.truncate else ''),
                            dataset_name,
                            config.window, config.step,
                            config.nb_layers,
                            config.nb_filters, config.k_size,
                            '_drop'+str(config.dropout) if config.dropout > 0 else '',
                            str(config.page), str(config.epochs),
                            str(config.batch))


# -------------------------------------------------------------------------
def get_cnn_logs_filename(folder, from_dataset, to_dataset, config):
    return '{}{}/logs_cnn_model_from_{}_to_{}_w{}_s{}_l{}_f{}_k{}_drop{}_page{}_e{}_b{}.npy'.format(
                            folder,
                            ('/truncated' if config.truncate else ''),
                            from_dataset, to_dataset,
                            config.window, config.step,
                            config.nb_layers,
                            config.nb_filters, config.k_size,
                            '_drop'+str(config.dropout) if config.dropout > 0 else '',
                            str(config.page), str(config.epochs),
                            str(config.batch))


# -------------------------------------------------------------------------
def label_smoothing_loss(y_true, y_pred):
    return tf.losses.sigmoid_cross_entropy(y_true, y_pred, label_smoothing=0.01)


# -------------------------------------------------------------------------
'''Create the source or label model separately'''
"""def build_source_model(input_shape, config):
    clsModel = utilModel.ModelSAE(input_shape, config)

    net = clsModel.get_model_features()
    net = clsModel.get_model_labels( net )
    label_model = Model(input=clsModel.input, output=net)

    # opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)

    model.compile(loss={'classifier_output': 'binary_crossentropy'},
                                               optimizer=opt, metrics=['mse'])

    return model"""


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
def train_cnn_batch(model, train_generator, batch_size):
    
    for batchXs, batchYs in train_generator:
        result = model.train_on_batch(batchXs, batchYs)

    #print(model.metrics_names)
    # print(result)
    return result


# ----------------------------------------------------------------------------
def __train_cnn_page(model, source_x_train, source_y_train, source_x_test, source_y_test,
                                                target_x_test, target_y_test,
                                                nb_epochs, batch_size, weights_filename, tensorboard):
    best_label_f1 = -1

    def named_logs(source_f1, target_f1):
        result = {}
        result["source_f1"] = source_f1
        result["target_f1"] = target_f1
        
        return result

    for e in range(nb_epochs):
        src_generator = batch_generator(source_x_train, source_y_train, batch_size=batch_size)

        # Train batch
        loss, label_mse = train_cnn_batch(model, src_generator, batch_size)

        source_prediction = model.predict(source_x_train, batch_size=32, verbose=0)
        source_f1, source_th = utilMetrics.calculate_best_fm(source_prediction, source_y_train)

        saved = ""
        if source_f1 <= best_label_f1:
            best_label_f1 = source_f1
            model.save(weights_filename)
            saved = "SAVED"

        target_loss, target_mse = model.evaluate(target_x_test, target_y_test, batch_size=32, verbose=0)

        y_pred = model.predict(target_x_test, batch_size=32, verbose=0)
        target_f1, target_th = utilMetrics.calculate_best_fm(y_pred, target_y_test)

        print("Epoch [{}/{}]: source label mse={:.4f}, f1={:.4f} | target label loss={:.4f}, mse={:.4f}, f1={:.4f} | {}".format(
                            e+1, nb_epochs, label_mse, source_f1, target_loss, target_mse, target_f1, saved))

        tensorboard.on_epoch_end(e, named_logs(source_f1=source_f1, target_f1=target_f1))

        """
        sample_images(
                                is_testing=False,
                                imgs_source=X_source_val[0:16],
                                labels_source=Y_source_val_16[0:16],
                                imgs_target=X_target_val[0:16],
                                labels_target=Y_target_val_16[0:16],
                                binarized_imgs=pred_binarized_target_val_argmax_16[0:16],
                                model_config=model_config,
                                mean_training_imgs=mean_training_imgs, std_training_imgs=std_training_imgs,
                                epoch = epoch)
        """

        gc.collect()


# ----------------------------------------------------------------------------
def train_cnn(model, source, target, page, nb_super_epoch, nb_epochs,
                                    batch_size, weights_filename, logs_filename, initial_hp_lambda=0.01):
    print('Training CNN model...')

    util.deleteFolder(logs_filename)

    tensorboard = TensorBoard(
                log_dir=logs_filename,
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_grads=True
                )
    tensorboard.set_model(model)

    for se in range(nb_super_epoch):
        print(80 * "-")
        print("SUPER EPOCH: %03d/%03d" % (se+1, nb_super_epoch))

        source['generator'].reset()
        source['generator'].shuffle()

        for source_x_train, source_y_train in source['generator']:
            print("> Train on %03d source page samples..." % (len(source_x_train)))
            #print(source_x_train.shape)
            #print(source_y_train.shape)

            __train_cnn_page(model, source_x_train, source_y_train, source['x_test'], source['y_test'],
                                                    target['x_test'], target['y_test'],
                                                    nb_epochs, batch_size, weights_filename, tensorboard)


"""
# ----------------------------------------------------------------------------
def train_cnn(model,  train_data_generator, source_x_test, source_y_test,
                                weights_filename, config):
    print('Fit CNN...')
    early_stopping = EarlyStopping(monitor='loss', patience=15)

    for se in range(config.nb_super_epoch):
        print(80 * "-")
        print("SUPER EPOCH: %03d/%03d" % (se+1, config.nb_super_epoch))

        train_data_generator.reset()
        train_data_generator.shuffle()

        for x_train, y_train in train_data_generator:
            print("> Train on %03d page samples..." % (len(x_train)))

            model.fit(x_train, y_train,
                                    batch_size=config.batch,
                                    epochs=config.epochs,
                                    verbose=2,
                                    shuffle=True,
                                    validation_data=(source_x_test, source_y_test),
                                    callbacks=[early_stopping])
            del x_train, y_train
            gc.collect()

            model.save_weights(weights_filename, overwrite=True)

    model.save_weights(weights_filename, overwrite=True)

    return model
"""