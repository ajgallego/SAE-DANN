# -*- coding: utf-8 -*-
import numpy as np
import argparse
import util, utilModels
import utilLoadMNIST, utilLoadMensural
from sklearn.utils import shuffle
from keras import backend as K
from keras.models import load_model


# ----------------------------------------------------------------------------
def __run_validations(datasets):
    input_shape = None
    nb_classes = None
    assert datasets is not None
    assert len(datasets) > 0

    for i in range(len(datasets)):
        assert 'name' in datasets[i] and datasets[i]['name'] is not None
        assert 'x_train' in datasets[i] and datasets[i]['x_train'] is not None
        assert 'y_train' in datasets[i] and datasets[i]['y_train'] is not None
        assert 'x_test' in datasets[i] and datasets[i]['x_test'] is not None
        assert 'y_test' in datasets[i] and datasets[i]['y_test'] is not None

        if input_shape is None:
            input_shape = datasets[i]['x_train'].shape[1:]
        assert input_shape == datasets[i]['x_train'].shape[1:]
        assert input_shape == datasets[i]['x_test'].shape[1:]

        if nb_classes is None:
            nb_classes = len(datasets[i]['y_train'][1])
        assert nb_classes == len(datasets[i]['y_train'][1])
        assert nb_classes == len(datasets[i]['y_test'][1])

    return input_shape, nb_classes


# ----------------------------------------------------------------------------
def train_dann(model_label, model_domain, source_x_train, source_y_train, target_x_train, target_y_train, weights, config):
    gen_source_batch = util.batch_generator([source_x_train, source_y_train], config.batch)  #// 2)
    gen_target_batch = util.batch_generator([target_x_train, target_y_train], config.batch // 2)

    imgs_per_epoch = source_x_train.shape[0]
    e = 0
    img_nr = 0
    best_label_acc = 0

    while e < config.epochs:
        X_s, y_s = next(gen_source_batch)
        X_t, __ = next(gen_target_batch)   # Target labels are not used

        X_d = np.vstack([X_s[0:config.batch // 2], X_t])    # Balance batch
        y_d = np.vstack([np.tile([1., 0.], [config.batch // 2, 1]), np.tile([0., 1.], [config.batch // 2, 1])])
        X_d, y_d = shuffle(X_d, y_d)

        label_loss, label_acc = model_label.train_on_batch(X_s, y_s)
        domain_loss, domain_acc = model_domain.train_on_batch(X_d, y_d)

        img_nr += config.batch
        saved = ""
        if img_nr > imgs_per_epoch:
            #img_nr -= imgs_per_epoch
            img_nr = 0
            e += 1
            if best_label_acc < label_acc:
                best_label_acc = label_acc
                model_label.save(weights)
                saved = "SAVED"

            target_loss, target_acc = model_label.evaluate(target_x_train, target_y_train, verbose=0)
            print("Epoch [{}/{}]: source label loss = {:.2f}, acc = {:.2f} | domain loss = {:.2f}, acc = {:.2f} | target label loss = {:.2f}, acc = {:.2f} | {}".format(
                            e, config.epochs, label_loss, label_acc, domain_loss, domain_acc, target_loss, target_acc, saved))


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DANN')
parser.add_argument('-db',        default='mensural', type=str, choices=['mnist', 'mensural'],  help='Dataset')
parser.add_argument('-mode',   default='dann', type=str, choices=['dann', 'cnn'],  help='Train type')
parser.add_argument('-fold',   default=-1,      type=int,    help='Select fold. -1 for all')
parser.add_argument('-e',      default=100,    type=int,   dest='epochs',         help='Number of epochs')
parser.add_argument('-b',      default=64,     type=int,   dest='batch',          help='Batch size')
parser.add_argument('-lda',      default=1.0,    type=float,    help='Reversal gradient lambda')
parser.add_argument('-lr1',      default=1.0,    type=float,    help='Label model learning rate')
parser.add_argument('-lr2',      default=1.0,    type=float,    help='Domain model learning rate')
parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')
args = parser.parse_args()

util.mkdirp('MODELS')

if args.db == 'mnist':
    datasets = utilLoadMNIST.load_datasets()
elif args.db == 'mensural':
    datasets = utilLoadMensural.load_datasets()
else:
    raise Exception('Unknowm dataset')

input_shape, num_labels = __run_validations(datasets)

num_domains = 2

print('CONFIG:')
print(' - DB:', args.db)
print(' - Input shape:', input_shape)
print(' - Num labels:', num_labels)
print(' - Num domains:', num_domains)
#print(' - Images per epoch:', datasets[0]['x_train'].shape[0])
print(' - Mode:', args.mode)
print(' - Epochs:', args.epochs)
print(' - Batch:', args.batch)
print(' - RG lambda:', args.lda)
print(' - Label learning rate:', args.lr1)
print(' - Domain learning rate:', args.lr2)


for i in range(len(datasets)):
    if args.fold != -1 and i != args.fold:
            continue

    for j in range(len(datasets)):
        if i == j:
            continue

        #if datasets[i]['name'] != 'BNE-BDH' and datasets[j]['name'] != 'BNE-BDH':                # TODO REMOVE...
        #    continue

        if args.db == 'mnist':
            model_label, model_domain = utilModels.mnist_model(input_shape, num_labels, num_domains, args)
        elif args.db == 'mensural':
            model_label, model_domain = utilModels.mensural_model(input_shape, num_labels, num_domains, args)
        else:
            raise Exception('Unknowm dataset')

        print(80*'-')
        print('SOURCE: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
            datasets[i]['name'], datasets[i]['x_train'].shape, datasets[i]['y_train'].shape,
            datasets[i]['x_test'].shape, datasets[i]['y_test'].shape))
        print('TARGET: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
            datasets[j]['name'], datasets[j]['x_train'].shape, datasets[j]['y_train'].shape,
            datasets[j]['x_test'].shape, datasets[j]['y_test'].shape))

        if args.mode == "dann":
            print('Train DANN')
            weights = 'MODELS/model_dann_from_{}_to_{}_best_label.h5'.format(datasets[i]['name'], datasets[j]['name'])
            train_dann(model_label, model_domain,
                                    datasets[i]['x_train'], datasets[i]['y_train'],
                                    datasets[j]['x_train'], datasets[j]['y_train'],
                                    weights, args)
            model_label = load_model(weights)

        elif args.mode == "cnn":
            print('Train CNN')
            model_label.fit(datasets[i]['x_train'], datasets[i]['y_train'],
                                    batch_size=args.batch,
                                    epochs=args.epochs,
                                    verbose=2,
                                    shuffle=True,
                                    validation_data=(datasets[j]['x_train'], datasets[j]['y_train']))

        # Final evaluation
        source_loss, source_acc = model_label.evaluate(datasets[i]['x_test'], datasets[i]['y_test'], verbose=0)
        target_loss, target_acc = model_label.evaluate(datasets[j]['x_test'], datasets[j]['y_test'], verbose=0)

        print('VALIDATION:')
        print('Result: {}\t{}\t{:.4f}\t{:.4f}'.format(datasets[i]['name'], datasets[j]['name'], source_acc, target_acc))
        #print(' - Source test set "{}" accuracy: {:.4f}'.format(datasets[i]['name'], source_acc))
        #print(' - Target test set "{}" accuracy: {:.4f}'.format(datasets[j]['name'], target_acc))


