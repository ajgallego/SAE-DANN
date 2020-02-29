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
    return '{}{}/weights_dannCONV_model_from_{}_to_{}_w{}_s{}_l{}_f{}_k{}_drop{}_page{}_e{}_b{}_lda{}.npy'.format(
                            folder,
                            ('/truncated' if config.truncate else ''),
                            from_dataset, to_dataset,
                            config.window, config.step,
                            config.nb_layers,
                            config.nb_filters, config.k_size,
                            '_drop'+str(config.dropout) if config.dropout > 0 else '',
                            str(config.page), str(config.epochs),
                            str(config.batch), str(config.lda))


def get_dann_logs_filename(folder, from_dataset, to_dataset, config):
    weights_filename = get_dann_weights_filename(folder, from_dataset, to_dataset, config)
    return weights_filename.replace("/weights_dann", "/logs_dann")

"""
# ----------------------------------------------------------------------------
def sample_images(self, is_testing, imgs_source, labels_source, imgs_target, labels_target, binarized_imgs, model_config, mean_training_imgs, std_training_imgs, epoch = None):
        r, c = 10, 8

        binarized_img3D = np.zeros((16, self.img_rows, self.img_cols, self.channels))
        labels_target_img3D = np.zeros((16, self.img_rows, self.img_cols, self.channels))
        labels_source_img3D = np.zeros((16, self.img_rows, self.img_cols, self.channels))
        idx = 0

        denorm_imgs_source = model_config.applyDeNormalization(imgs_source[0:16], mean_training_imgs, std_training_imgs)
        denorm_imgs_target = model_config.applyDeNormalization(imgs_target[0:16], mean_training_imgs, std_training_imgs)
        denorm_binarized_imgs = (1-binarized_imgs[0:16,:,:]) * 255
        denorm_labels_target = (1-labels_target[0:16,:,:]) * 255
        denorm_labels_source = (1-labels_source[0:16,:,:]) * 255

        for channel in range(self.channels):
            binarized_img3D[:,:,:,channel] = denorm_binarized_imgs.astype(np.uint8)
            labels_target_img3D[:,:,:,channel] = denorm_labels_target.astype(np.uint8)
            labels_source_img3D[:,:,:,channel] = denorm_labels_source.astype(np.uint8)
            idx = idx + 1

        gen_imgs = np.concatenate([denorm_imgs_source, labels_source_img3D, denorm_imgs_target, labels_target_img3D, binarized_img3D])
        gen_imgs = gen_imgs.astype(np.uint8)

        #titles = ['Original', 'Translated']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                im = gen_imgs[cnt,:,:,:]
                axs[i,j].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                #axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1

        if is_testing:
            pathdir = self.pathdir_testing_images.replace(".", "-")
        else:
            pathdir = self.pathdir_training_images.replace(".", "-")

        util.mkdirp(pathdir)

        if (epoch is None):
            fig.savefig("%s/SAE.png" % (pathdir))
        else:
            fig.savefig("%s/%d.png" % (pathdir, epoch))
        plt.close()
"""


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
                                                nb_epochs, batch_size, weights_filename, tensorboard):
    best_label_f1 = -1
    target_genenerator = batch_generator(target_x_train, None, batch_size=batch_size // 2)

    def named_logs(source_f1, target_f1, hp_lambda):
        result = {}
        result["source_f1"] = source_f1
        result["target_f1"] = target_f1
        result["lambda"] = hp_lambda
        
        return result

    for e in range(nb_epochs):
        src_generator = batch_generator(source_x_train, source_y_train, batch_size=batch_size // 2)

        # Update learning rates
        if type(dann_builder.opt) is str:
            lr = dann_builder.opt
        else:
            lr = float(K.get_value(dann_builder.opt.lr))* (1. / (1. + float(K.get_value(dann_builder.opt.decay)) * float(K.get_value(dann_builder.opt.iterations)) ))
        print(' - Lr:', lr, ' / Lambda:', dann_builder.grl_layer.get_hp_lambda())

        dann_builder.grl_layer.increment_hp_lambda_by(1e-4)      #1e-6  1e-4)  # !!!  ### NEW MODEL ####

        # Train batch
        loss, domain_loss, label_loss, domain_acc, label_mse = train_dann_batch(
                                            dann_builder.dann_model, src_generator, target_genenerator, target_x_train, batch_size )

        source_prediction = dann_builder.label_model.predict(source_x_train, batch_size=32, verbose=0)
        source_f1, source_th = utilMetrics.calculate_best_fm(source_prediction, source_y_train)

        saved = ""
        if source_f1 <= best_label_f1:
            best_label_f1 = source_f1
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

        tensorboard.on_epoch_end(e, named_logs(source_f1=source_f1, target_f1=target_f1, hp_lambda=dann_builder.grl_layer.get_hp_lambda()))


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
def train_dann(dann_builder, source, target, page, nb_super_epoch, nb_epochs,
                                    batch_size, weights_filename, logs_filename, initial_hp_lambda=0.01):
    print('Training DANN model...')

    dann_builder.grl_layer.set_hp_lambda(initial_hp_lambda)

    util.deleteFolder(logs_filename)
    tensorboard = TensorBoard(
                log_dir=logs_filename,
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_grads=True
                )
    tensorboard.set_model(dann_builder.dann_model)

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
                                                    nb_epochs, batch_size, weights_filename, tensorboard)

