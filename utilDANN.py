# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import util
import utilMetrics


# ----------------------------------------------------------------------------
def get_dann_weights_filename(folder, from_dataset, to_dataset, config):
    if config.grl_position == 0:
        grl_position_str = ""
    else:
        grl_position_str = "_gpos" + str(config.grl_position)

    return '{}{}/weights_dannCONV_model_from_{}_to_{}_w{}_s{}_l{}_f{}_k{}_drop{}_page{}_super{}_e{}_b{}_lda{}_lda_inc{}_dmodel{}{}.npy'.format(
                            folder,
                            ('/truncated' if config.truncate else ''),
                            from_dataset, to_dataset,
                            config.window, config.step,
                            config.nb_layers,
                            config.nb_filters, config.k_size,
                            str(config.dropout),
                            str(config.page), str(config.nb_super_epoch),
                            str(config.epochs), str(config.batch),
                            str(config.lda), str(config.lda_inc),
                            str(config.domain_model_version),
                            grl_position_str)

# ----------------------------------------------------------------------------
def get_dann_logs_directory(folder, from_dataset, to_dataset, config):
    weights_filename = get_dann_weights_filename(folder, from_dataset, to_dataset, config)
    return weights_filename.replace("/weights_dann", "/logs_dann")

# ----------------------------------------------------------------------------
def get_dann_csv_logs_directory(folder, from_dataset, to_dataset, config):
    weights_filename = get_dann_weights_filename(folder, from_dataset, to_dataset, config)
    return weights_filename.replace("/weights_dann", "/csv_logs_dann")

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

    domain_output_shape = dann_model.output_shape[0]
    #print(domain_output_shape)
    batchYd = np.concatenate(( np.zeros((batch_size // 2,) + domain_output_shape[1:], dtype=int),
                                                              np.ones((batch_size // 2,) + domain_output_shape[1:], dtype=int)) )
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
                                                nb_epochs, batch_size,
                                                lda_inc,
                                                weights_filename,
                                                csv_logs_directory,
                                                page,
                                                with_tensorboard, tensorboard):
    best_label_f1 = -1
    target_genenerator = batch_generator(target_x_train, None, batch_size=batch_size // 2)

    def named_logs(source_f1_train, source_f1_test, target_f1_train, target_f1_test, hp_lambda):
        result = {}
        result["source_f1_train"] = source_f1_train
        result["source_f1_test"] = source_f1_test
        result["target_f1_train"] = target_f1_train
        result["target_f1_test"] = target_f1_test
        result["lambda"] = hp_lambda
        return result

    util.mkdirp(csv_logs_directory)
    csv_logs_filename = csv_logs_directory + "/" + str(page) + "_logs.csv"
    csv_logs_file = open(csv_logs_filename,'a+')
    csv_logs_file.write("#Epoch\tSRC_TRAIN\tTRG_TRAIN\tSRC_TEST\tTRG_TEST\tTRG_MSE\tTRG_LOSS\tLBL_MSE\tLBL_LOSS\tLAMBDA\n")
    csv_logs_file.close()

    for e in range(nb_epochs):
        src_generator = batch_generator(source_x_train, source_y_train, batch_size=batch_size // 2)

        # Update learning rates
        if type(dann_builder.opt) is str:
            lr = dann_builder.opt
        else:
            lr = float(K.get_value(dann_builder.opt.lr))* (1. / (1. + float(K.get_value(dann_builder.opt.decay)) * float(K.get_value(dann_builder.opt.iterations)) ))
        print(' - Lr:', lr, ' / Lambda:', dann_builder.grl_layer.get_hp_lambda())

        dann_builder.grl_layer.increment_hp_lambda_by(lda_inc)

        # Train batch
        logs = train_dann_batch(
                                            dann_builder.dann_model, src_generator, target_genenerator, target_x_train, batch_size )

        loss, domain_loss, label_loss, domain_acc, label_mse = logs

        print("Source train")
        source_prediction_train = dann_builder.label_model.predict(source_x_train, batch_size=32, verbose=0)
        source_f1_train, source_th_train = utilMetrics.calculate_best_fm(source_prediction_train, source_y_train)

        print("Source test")
        source_prediction_test = dann_builder.label_model.predict(source_x_test, batch_size=32, verbose=0)
        source_f1_test, source_th_test = utilMetrics.calculate_best_fm(source_prediction_test, source_y_test)

        #print("Target train")
        #target_prediction_train = dann_builder.label_model.predict(target_x_train, batch_size=32, verbose=0)
        #target_f1_train, target_th_train = utilMetrics.calculate_best_fm(target_prediction_train, target_y_train)

        print("Target test")
        target_prediction_test = dann_builder.label_model.predict(target_x_test, batch_size=32, verbose=0)
        target_f1_test, target_th_test = utilMetrics.calculate_best_fm(target_prediction_test, target_y_test, source_th_test)

        saved = ""
        if source_f1_test >= best_label_f1:
            best_label_f1 = source_f1_test
            dann_builder.save(weights_filename)
            saved = "SAVED"

        #target_loss, target_mse = dann_builder.label_model.evaluate(target_x_train, target_y_train, batch_size=32, verbose=0)
        target_loss, target_mse = dann_builder.label_model.evaluate(target_x_test, target_y_test, batch_size=32, verbose=0)

        #print("Epoch [{}/{}]: source label loss={:.4f}, mse={:.4f} | domain loss={:.4f}, acc={:.4f} | target label loss={:.4f}, mse={:.4f} | {}".format(
        #                    e+1, nb_epochs, label_loss, label_mse, domain_loss, domain_acc, target_loss, target_mse, saved))
        print("Epoch [{}/{}]: source label loss={:.4f}, mse={:.4f}, f1={:.4f} | domain loss={:.4f}, acc={:.4f} | target label loss={:.4f}, mse={:.4f}, f1={:.4f} | {}".format(
                            e+1, nb_epochs, label_loss, label_mse, source_f1_train, domain_loss, domain_acc, target_loss, target_mse, target_f1_test, saved))

        if with_tensorboard:
            tensorboard.on_epoch_end(e, named_logs(
                                source_f1_train=source_f1_train,
                                source_f1_test=source_f1_test,
                                target_f1_train=0, #target_f1_train,
                                target_f1_test=target_f1_test,
                                hp_lambda=dann_builder.grl_layer.get_hp_lambda()))

        csv_logs_file = open(csv_logs_filename,'a+')
        csv_logs_file.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%\
                                (e,\
                                source_f1_train,\
                                0,\
                                #target_f1_train,\
                                source_f1_test,\
                                target_f1_test,\
                                target_mse,\
                                target_loss,\
                                label_mse,\
                                label_loss,\
                                dann_builder.grl_layer.get_hp_lambda()\
                                ))
        csv_logs_file.close()

        gc.collect()



# ----------------------------------------------------------------------------
def train_dann(dann_builder, source, target,
                                    weights_filename, parent_logs_directory, csv_logs_directory,
                                    config):
    print('Training DANN model...')

    dann_builder.grl_layer.set_hp_lambda(config.lda)

    logs_directory = get_dann_logs_directory( parent_logs_directory, source['name'], target['name'], config)
    util.deleteFolder(logs_directory)

    csv_logs_directory = get_dann_csv_logs_directory( csv_logs_directory, source['name'], target['name'], config)
    util.deleteFolder(csv_logs_directory)

    if config.tboard:
        tensorboard = TensorBoard(
                log_dir=logs_directory,
                histogram_freq=0,
                batch_size=config.batch,
                write_graph=True,
                write_grads=True
                )
        tensorboard.set_model(dann_builder.dann_model)
    else:
        tensorboard = None

    for se in range(config.nb_super_epoch):
        print(80 * "-")
        print("SUPER EPOCH: %03d/%03d" % (se+1, config.nb_super_epoch))

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

            page = source['generator'].get_pos()

            __train_dann_page(dann_builder, source_x_train, source_y_train, source['x_test'], source['y_test'],
                                                    target_x_train, target_y_train, target['x_test'], target['y_test'],
                                                    config.epochs, config.batch,
                                                    config.lda_inc,
                                                    weights_filename,
                                                    csv_logs_directory,
                                                    page,
                                                    config.tboard, tensorboard)

