# -*- coding: utf-8 -*-
#import utilModels
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


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


