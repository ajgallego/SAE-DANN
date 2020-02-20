#!/usr/bin/env python
# -*- coding: utf-8 -*-
import abc
from keras import layers, regularizers, initializers, optimizers
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.layers import Input, Dropout, Activation, MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate, Subtract
import keras.backend as K


# ----------------------------------------------------------------------------
class AbstractModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_shape):
        self.input = Input(shape=input_shape, name="main_input")

    @abc.abstractmethod
    def get_model_features(self):
        pass

    @abc.abstractmethod
    def get_model_labels(self, input):
        pass

    @abc.abstractmethod
    def get_model_domains(self, input):
        pass


# ----------------------------------------------------------------------------
class ModelSAE(AbstractModel):
    # -----------------------------------------
    def __init__(self, input_shape, config):
        AbstractModel.__init__(self, input_shape)
        self.config = config
        self.config.strides = 2
        self.bn_axis = self.__get_normalization_axis()
        self.encoderLayers = [None] * self.config.nb_layers

    # -----------------------------------------
    def __get_normalization_axis(self):
        if K.image_data_format() == 'channels_last':
            return 3
        return 1

    # -----------------------------------------
    def __create_layer_conv(self, from_layer, deconv=False):
        kernel_initializer = initializers.glorot_uniform(seed=42)   # zeros  glorot_uniform  glorot_normal lecun_normal
        kernel_regularizer = regularizers.l2(0.01)  # None, 0.01
        activity_regularizer = None  # regularizers.l1(0.01)
        if deconv is not True:
            x = Conv2D( self.config.nb_filters, kernel_size=self.config.k_size, strides=self.config.strides,
                                    kernel_initializer=kernel_initializer,
                                    kernel_regularizer = kernel_regularizer,
                                    activity_regularizer = activity_regularizer,
                                    padding='same')(from_layer)
        else:
            x = Conv2DTranspose(self.config.nb_filters, kernel_size=self.config.k_size, strides=self.config.strides,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer = kernel_regularizer,
                                activity_regularizer = activity_regularizer,
                                padding='same')(from_layer)
        x = BatchNormalization(axis=self.bn_axis)(x)
        x = Activation('relu')(x)
        if self.config.dropout > 0:
            x = Dropout(self.config.dropout, seed=42)(x)
        return x

    # -----------------------------------------
    def get_model_features(self):
        x = self.input
        for i in xrange(self.config.nb_layers):
            x = self.__create_layer_conv(x)
            self.encoderLayers[i] = x
        return x

    # -----------------------------------------
    def get_model_labels(self, input):
        x = input
        for i in xrange(self.config.nb_layers):
            x = self.__create_layer_conv(x, True)
            ind = self.config.nb_layers - i - 2
            if ind >= 0:
                x = layers.add([x, self.encoderLayers[ind]])
        x = Conv2D(1, kernel_size=self.config.k_size, strides=1,
                                    kernel_initializer = initializers.glorot_uniform(seed=42),   # 'glorot_uniform', # zeros
                                    kernel_regularizer = None,
                                    activity_regularizer = None,
                                    name='classifier_output',           #'features_inc',
                                    padding='same', activation='sigmoid')(x)
        return x

    # -----------------------------------------
    def get_model_domains(self, input):
        #x = Flatten()(input)
        x = GlobalAveragePooling2D()(input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(2, activation='softmax', name='domain_output')(x)
        return x

