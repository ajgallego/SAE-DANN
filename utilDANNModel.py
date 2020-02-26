# -*- coding: utf-8 -*-
import utilModel
import numpy as np
from utilGradientReversal import GradientReversal
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Lambda
from keras import backend as K


class DANNModel(object):
    # -------------------------------------------------------------------------
    def __init__(self, input_shape, config, summary=False):
        self.learning_phase = K.variable(1)
        self.input_shape = input_shape
        self.batch_size = config.batch
        self.summary = summary

        # Default:      optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # opt = optimizers.Adam(lr=0.001, decay=0.01)  # 0.005

        self.opt = 'adam' #
        #self.opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        self.clsModel = utilModel.ModelSAE(input_shape, config)

        self.dann_model, self.label_model, self.tsne_model = self.__build_dann_model()

        self.__compile()

    # -------------------------------------------------------------------------
    def load(self, filename):
        weight = np.load(filename, allow_pickle=True)
        self.dann_model.set_weights(weight)

    # -------------------------------------------------------------------------
    def save(self, filename):
        np.save(filename, self.dann_model.get_weights())

    # -------------------------------------------------------------------------
    def __compile(self):
        #### NEW MODEL ####
        # self.dann_model.compile(loss={'classifier_output': 'binary_crossentropy',
        #                                                                    'domain_output': 'binary_crossentropy'},  ###
        self.dann_model.compile(loss={'classifier_output': 'binary_crossentropy',
                                                                            'domain_output': 'binary_crossentropy'},
                                                               loss_weights={'classifier_output': 0.5, 'domain_output': 1.0},
                                                               optimizer=self.opt,
                                                               metrics={'classifier_output': 'mse',
                                                                                    'domain_output': 'accuracy'})
                                                               #metrics=['mse'])

        self.label_model.compile(loss='binary_crossentropy',
                                                                 optimizer=self.opt,
                                                                 metrics=['mse'])

        self.tsne_model.compile(loss='binary_crossentropy',
                                                                optimizer=self.opt,
                                                                metrics=['mse'])


    # -------------------------------------------------------------------------
    def __build_dann_model(self):
        branch_features = self.clsModel.get_model_features()

        # Build domain model...
        self.grl_layer = GradientReversal(1.0)
        branch_domain = self.grl_layer(branch_features)
        branch_domain = self.clsModel.get_model_domains(branch_domain)

        # Build label model...
        # When building DANN model, route first half of batch (source examples)
        # to domain classifier, and route full batch (half source, half target)
        # to the domain classifier.
        branch_label = Lambda(lambda x: K.switch(K.learning_phase(),
                                                                                                 K.concatenate([x[:int(self.batch_size//2)],
                                                                                                                                   x[:int(self.batch_size//2)]], axis=0),
                                                                                                 x),
                                                        output_shape=lambda x: x[0:])(branch_features)

        # Build label model...
        branch_label = self.clsModel.get_model_labels(branch_label)

        # Create models...
        dann_model = Model(input=self.clsModel.input, output=[branch_domain, branch_label])
        label_model = Model(input=self.clsModel.input, output=branch_label)
        tsne_model = Model(input=self.clsModel.input, output=branch_features)

        if self.summary:
            print(dann_model.summary())

        return dann_model, label_model, tsne_model

