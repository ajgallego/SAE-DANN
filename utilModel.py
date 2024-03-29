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

    def __init__(self, input_shape, domain_model_version):
        self.input = Input(shape=input_shape, name="main_input")
        self.domain_model_version = domain_model_version

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
        AbstractModel.__init__(self, input_shape, config.domain_model_version)
        self.config = config
        self.config.strides = 2
        self.bn_axis = self.__get_normalization_axis()
        self.encoderLayers = [None] * self.config.nb_layers

        grl_position_respect_latent_code = self.config.grl_position
        self.grl_position_respect_global = self.config.nb_layers + grl_position_respect_latent_code
        assert(self.grl_position_respect_global >= 0 and self.grl_position_respect_global <= self.config.nb_layers*2)

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
        # x = BatchNormalization(axis=self.bn_axis)(x)          #### NEW MODEL ####
        x = Activation('relu')(x)
        if self.config.dropout > 0:                                                     #### NEW MODEL ####
            x = Dropout(self.config.dropout, seed=42)(x)
        return x

    # -----------------------------------------
    def get_model_features(self):
        x = self.input

        for i in xrange(min(self.grl_position_respect_global, self.config.nb_layers)):
            x = self.__create_layer_conv(x)
            self.encoderLayers[i] = x

        for i in xrange(abs(self.config.nb_layers - max(self.grl_position_respect_global, self.config.nb_layers))):
            x = self.__create_layer_conv(x, True)
            ind = self.config.nb_layers - i - 2
            if ind >= 0:
                x = layers.add([x, self.encoderLayers[ind]])

        return x


    # -----------------------------------------
    def __get_model_labels(
                        self, 
                        input, 
                        nb_layers, 
                        k_size,
                        grl_position_respect_global, 
                        tag,
                        with_residual_connections=False):
        
        x = input

        for i in xrange(nb_layers - min(grl_position_respect_global, nb_layers)):
            x = self.__create_layer_conv(x)
            self.encoderLayers[min(grl_position_respect_global, nb_layers) + i] = x

        previous_dec_layers = abs(self.config.nb_layers - max(self.grl_position_respect_global, self.config.nb_layers))
        for i in xrange(nb_layers - abs(nb_layers - max(grl_position_respect_global, nb_layers))):
            x = self.__create_layer_conv(x, True)
            ind = nb_layers - previous_dec_layers - i - 2
            if with_residual_connections and ind >= 0:
                x = layers.add([x, self.encoderLayers[ind]])

        x = Conv2D(1, kernel_size=k_size, strides=1,
                                    kernel_initializer = initializers.glorot_uniform(seed=42),   # 'glorot_uniform', # zeros
                                    kernel_regularizer = None,
                                    activity_regularizer = None,
                                    name=tag,           #'features_inc',
                                    padding='same', activation='sigmoid')(x)
        return x

    # -----------------------------------------
    def get_model_labels(self, input):

        return self.__get_model_labels(
                        input, 
                        self.config.nb_layers, 
                        self.config.k_size,
                        self.grl_position_respect_global, 
                        'classifier_output',
                        True)

    def model_domain_v1(self, input):
        #x = Flatten()(input)
        x = GlobalAveragePooling2D()(input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(2, activation='softmax', name='domain_output')(x)

        return x

    def model_domain_v2(self, input):
        #### NEW MODEL ####
        back = self.config.nb_filters
        self.config.nb_filters = int(back / 4)

        x = self.__get_model_labels(
                        input, 
                        self.config.nb_layers, 
                        self.config.k_size,
                        self.grl_position_respect_global, 
                        'domain_output',
                        False)

        self.config.nb_filters = back

        return x

    # -----------------------------------------
    def get_model_domains(self, input):

        if self.domain_model_version == 1:
            return self.model_domain_v1(input)
        elif self.domain_model_version == 2:
            return self.model_domain_v2(input)
        else:
            assert(False)
        
