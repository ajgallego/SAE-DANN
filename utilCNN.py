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
    return '{}{}/weights_cnn_model_{}_w{}_s{}_l{}_f{}_k{}_drop{}_page{}_super{}_e{}_b{}.npy'.format(
                            folder,
                            ('/truncated' if config.truncate else ''),
                            dataset_name,
                            config.window, config.step,
                            config.nb_layers,
                            config.nb_filters, config.k_size,
                            '_drop'+str(config.dropout) if config.dropout > 0 else '',
                            str(config.page), str(config.nb_super_epoch),
                            str(config.epochs), str(config.batch))


# -------------------------------------------------------------------------
def get_cnn_logs_directory(folder, from_dataset, to_dataset, config):
    filename = get_cnn_weights_filename(folder, from_dataset, config)
    filename = filename.replace('/weights_cnn_model_', '/logs_cnn_model_from_')
    filename = filename.replace('_w', '_to'+to_dataset+'_w')
    return filename

    """return '{}{}/logs_cnn_model_from_{}_to_{}_w{}_s{}_l{}_f{}_k{}_drop{}_page{}_super{}_e{}_b{}.npy'.format(
                            folder,
                            ('/truncated' if config.truncate else ''),
                            from_dataset, to_dataset,
                            config.window, config.step,
                            config.nb_layers,
                            config.nb_filters, config.k_size,
                            '_drop'+str(config.dropout) if config.dropout > 0 else '',
                            str(config.page), str(config.nb_super_epoch),
                            str(config.epochs), str(config.batch))"""


# -------------------------------------------------------------------------
def get_cnn_csv_logs_directory(folder, from_dataset, to_dataset, config):
    weights_filename = get_cnn_logs_directory(folder, from_dataset, to_dataset, config)
    return weights_filename.replace("/logs_cnn_model", "/csv_logs_cnn_model")


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
                                                target_x_train, target_y_train,
                                                target_x_test, target_y_test,
                                                nb_epochs, batch_size,
                                                weights_filename,
                                                csv_logs_directory,
                                                page,
                                                with_tensorboard, tensorboard):
    best_label_f1 = -1

    def named_logs(source_f1_train, source_f1_test, target_f1_train, target_f1_test):
        result = {}
        result["source_f1_train"] = source_f1_train
        result["source_f1_test"] = source_f1_test
        result["target_f1_train"] = target_f1_train
        result["target_f1_test"] = target_f1_test
        return result

    util.mkdirp(csv_logs_directory)
    csv_logs_filename = csv_logs_directory + "/" + str(page) + "_logs.csv"
    csv_logs_file = open(csv_logs_filename,'a+')
    csv_logs_file.write("#Epoch\tSRC_TRAIN\tTRG_TRAIN\tSRC_TEST\tTRG_TEST\tTRG_MSE\tTRG_LOSS\tLBL_MSE\tLBL_LOSS\n")
    csv_logs_file.close()

    for e in range(nb_epochs):
        src_generator = batch_generator(source_x_train, source_y_train, batch_size=batch_size)

        # Train batch
        loss, label_mse = train_cnn_batch(model, src_generator, batch_size)

        source_prediction_train = model.predict(source_x_train, batch_size=32, verbose=0)
        source_f1_train, source_th_train = utilMetrics.calculate_best_fm(source_prediction_train, source_y_train)

        source_prediction_test = model.predict(source_x_test, batch_size=32, verbose=0)
        source_f1_test, source_th_test = utilMetrics.calculate_best_fm(source_prediction_test, source_y_test)

        target_prediction_train = model.predict(target_x_train, batch_size=32, verbose=0)
        target_f1_train, target_th_train = utilMetrics.calculate_best_fm(target_prediction_train, target_y_train)

        target_prediction_test = model.predict(target_x_test, batch_size=32, verbose=0)
        target_f1_test, target_th_test = utilMetrics.calculate_best_fm(target_prediction_test, target_y_test)

        saved = ""
        if source_f1_test >= best_label_f1:
            best_label_f1 = source_f1_test
            model.save_weights(weights_filename)
            saved = "SAVED"

        target_loss, target_mse = model.evaluate(target_x_test, target_y_test, batch_size=32, verbose=0)

        print("Epoch [{}/{}]: source label mse={:.4f}, f1={:.4f} | target label loss={:.4f}, mse={:.4f}, f1={:.4f} | {}".format(
                            e+1, nb_epochs, label_mse, source_f1_test, target_loss, target_mse, target_f1_test, saved))

        if with_tensorboard:
            tensorboard.on_epoch_end(e, named_logs(
                                source_f1_train=source_f1_train,
                                source_f1_test=source_f1_test,
                                target_f1_train=target_f1_train,
                                target_f1_test=target_f1_test))

        csv_logs_file = open(csv_logs_filename,'a+')
        csv_logs_file.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%\
                                (e,\
                                source_f1_train,\
                                target_f1_train,\
                                source_f1_test,\
                                target_f1_test,\
                                target_mse,\
                                target_loss,\
                                label_mse
                                ))
        csv_logs_file.close()

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
def train_cnn(model, source, target,
                    weights_filename, parent_logs_directory, parent_csv_logs_directory, config):
    print('Training CNN model...')

    csv_logs_directory = get_cnn_csv_logs_directory( parent_csv_logs_directory, source['name'], target['name'], config)
    util.deleteFolder(csv_logs_directory)

    logs_directory = get_cnn_logs_directory( parent_logs_directory, source['name'], target['name'], config)
    util.deleteFolder(logs_directory)
    tensorboard = TensorBoard(
                log_dir=logs_directory,
                histogram_freq=0,
                batch_size=config.batch,
                write_graph=True,
                write_grads=True
                )
    tensorboard.set_model(model)

    for se in range(config.nb_super_epoch):
        print(80 * "-")
        print("SUPER EPOCH: %03d/%03d" % (se+1, config.nb_super_epoch))

        source['generator'].reset()
        source['generator'].shuffle()

        for source_x_train, source_y_train in source['generator']:
            print("> Train on %03d source page samples..." % (len(source_x_train)))
            #print(source_x_train.shape)
            #print(source_y_train.shape)

            try:
                target_x_train, target_y_train = next(target['generator'])
            except: # Restart...
                target['generator'].reset()
                target['generator'].shuffle()
                target_x_train, target_y_train = next(target['generator'])

            page = source['generator'].get_pos()

            __train_cnn_page(model, source_x_train, source_y_train, source['x_test'], source['y_test'],
                                                    target_x_train, target_y_train,
                                                    target['x_test'], target['y_test'],
                                                    config.epochs, config.batch, weights_filename,
                                                    csv_logs_directory,
                                                    page,
                                                    config.tboard, tensorboard)

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