# -*- coding: utf-8 -*-
import os
import util
import numpy as np

ARRAY_DBS =['dibco2016','dibco2014','palm0','palm1','phi','ein','sal','voy','bdi','all',
            'dibco2016-ic','dibco2014-ic','palm0-ic','palm1-ic','phi-ic','ein-ic','sal-ic','voy-ic','bdi-ic','all-ic',
            'sal-oe10',
            'sal-oe0.4',
            'sal-blur30x30']


# -----------------------------------------------------------------------------
def load_array_of_files(basepath, folders, truncate=False):
    X = []
    for folder in folders:
        full_path = os.path.join(basepath, folder)
        array_of_files = util.list_files(full_path, ext='png')

        for fname_x in array_of_files:
            X.append(fname_x)

    if truncate:
        X = X[:10]

    return np.asarray(X)


# ----------------------------------------------------------------------------
def load_folds_names(dbname):
    assert dbname in ARRAY_DBS

    train_folds = []
    test_folds = []

    DIBCO = [    ['Dibco/2009/handwritten_GR', 'Dibco/2009/printed_GR'],
                 ['Dibco/2010/handwritten_GR'],
                 ['Dibco/2011/handwritten_GR', 'Dibco/2011/printed_GR'],
                 ['Dibco/2012/handwritten_GR'],
                 ['Dibco/2013/handwritten_GR', 'Dibco/2013/printed_GR'],
                 ['Dibco/2014/handwritten_GR'],
                 ['Dibco/2016/handwritten_GR']     ]

    DIBCO_synthetic_inv_col = [
                 ['synthetic/inv_col/Dibco/2009/handwritten_GR', 'synthetic/inv_col/Dibco/2009/printed_GR'],
                 ['synthetic/inv_col/Dibco/2010/handwritten_GR'],
                 ['synthetic/inv_col/Dibco/2011/handwritten_GR', 'synthetic/inv_col/Dibco/2011/printed_GR'],
                 ['synthetic/inv_col/Dibco/2012/handwritten_GR'],
                 ['synthetic/inv_col/Dibco/2013/handwritten_GR', 'synthetic/inv_col/Dibco/2013/printed_GR'],
                 ['synthetic/inv_col/Dibco/2014/handwritten_GR'],
                 ['synthetic/inv_col/Dibco/2016/handwritten_GR']     ]

    PALM_train = [ ['Palm/Challenge-1-ForTrain/gt1_GR'], ['Palm/Challenge-1-ForTrain/gt2_GR'] ]
    PALM_test = [ ['Palm/Challenge-1-ForTest/gt1_GR'], ['Palm/Challenge-1-ForTest/gt2_GR'] ]

    PALM_train_synthetic_inv_col = [ ['synthetic/inv_col/Palm/Challenge-1-ForTrain/gt1_GR'], ['synthetic/inv_col/Palm/Challenge-1-ForTrain/gt2_GR'] ]
    PALM_test_synthetic_inv_col = [ ['synthetic/inv_col/Palm/Challenge-1-ForTest/gt1_GR'], ['synthetic/inv_col/Palm/Challenge-1-ForTest/gt2_GR'] ]

    PHI_train = ['PHI/train/phi_GR']
    PHI_test = ['PHI/test/phi_GR']

    PHI_train_synthetic_inv_col = ['synthetic/inv_col/PHI/train/phi_GR']
    PHI_test_synthetic_inv_col = ['synthetic/inv_col/PHI/test/phi_GR']

    EINSIELDELN_train = ['Einsieldeln/train/ein_GR']
    EINSIELDELN_test = ['Einsieldeln/test/ein_GR']

    EINSIELDELN_train_synthetic_inv_col = ['synthetic/inv_col/Einsieldeln/train/ein_GR']
    EINSIELDELN_test_synthetic_inv_col = ['synthetic/inv_col/Einsieldeln/test/ein_GR']

    SALZINNES_train = ['Salzinnes/train/sal_GR']
    SALZINNES_test = ['Salzinnes/test/sal_GR']

    SALZINNES_train_synthetic_overexposure10 = ['synthetic/overexposure_g10/Salzinnes/train/sal_GR']
    SALZINNES_test_synthetic_overexposure10 = ['synthetic/overexposure_g10/Salzinnes/test/sal_GR']

    SALZINNES_train_synthetic_overexposure0_4 = ['synthetic/overexposure_g0.4/Salzinnes/train/sal_GR']
    SALZINNES_test_synthetic_overexposure0_4 = ['synthetic/overexposure_g0.4/Salzinnes/test/sal_GR']

    SALZINNES_train_synthetic_blur30x30 = ['synthetic/blur_30x30/Salzinnes/train/sal_GR']
    SALZINNES_test_synthetic_blur30x30 = ['synthetic/blur_30x30/Salzinnes/test/sal_GR']
    
    SALZINNES_train_synthetic_inv_col = ['synthetic/inv_col/Salzinnes/train/sal_GR']
    SALZINNES_test_synthetic_inv_col = ['synthetic/inv_col/Salzinnes/test/sal_GR']

    VOYNICH_test = ['Voynich/voy_GR']
    VOYNICH_test_synthetic_inv_col = ['synthetic/inv_col/Voynich/voy_GR']

    BDI_train = ['BDI/train/bdi11_GR']
    BDI_test = ['BDI/test/bdi11_GR']

    BDI_train_synthetic_inv_col = ['synthetic/inv_col/BDI/train/bdi11_GR']
    BDI_test_synthetic_inv_col = ['synthetic/inv_col/BDI/test/bdi11_GR']


    if dbname == 'dibco2016':
        test_folds = DIBCO[6]
        DIBCO.pop(6)
        train_folds = [val for sublist in DIBCO for val in sublist]
    elif dbname == 'dibco2016-ic':
        test_folds = DIBCO_synthetic_inv_col[6]
        DIBCO_synthetic_inv_col.pop(6)
        train_folds = [val for sublist in DIBCO_synthetic_inv_col for val in sublist]
    elif dbname == 'dibco2014':
        test_folds = DIBCO[5]
        DIBCO.pop(5)
        train_folds = [val for sublist in DIBCO for val in sublist]
    elif dbname == 'dibco2014-ic':
        test_folds = DIBCO_synthetic_inv_col[5]
        DIBCO_synthetic_inv_col.pop(5)
        train_folds = [val for sublist in DIBCO_synthetic_inv_col for val in sublist]
    elif dbname == 'palm0':
        train_folds = PALM_train[0]
        test_folds = PALM_test[0]
    elif dbname == 'palm0-ic':
        train_folds = PALM_train_synthetic_inv_col[0]
        test_folds = PALM_test_synthetic_inv_col[0]
    elif dbname == 'palm1':
        train_folds = PALM_train[1]
        test_folds = PALM_test[1]
    elif dbname == 'palm1-ic':
        train_folds = PALM_train_synthetic_inv_col[1]
        test_folds = PALM_test_synthetic_inv_col[1]
    elif dbname == 'phi':
        train_folds = PHI_train
        test_folds = PHI_test
    elif dbname == 'phi-ic':
        train_folds = PHI_train_synthetic_inv_col
        test_folds = PHI_test_synthetic_inv_col
    elif dbname == 'ein':
        train_folds = EINSIELDELN_train
        test_folds = EINSIELDELN_test
    elif dbname == 'ein-ic':
        train_folds = EINSIELDELN_train_synthetic_inv_col
        test_folds = EINSIELDELN_test_synthetic_inv_col
    elif dbname == 'sal':
        train_folds = SALZINNES_train
        test_folds = SALZINNES_test
    elif dbname == 'sal-ic':
        train_folds = SALZINNES_train_synthetic_inv_col
        test_folds = SALZINNES_test_synthetic_inv_col
    elif dbname == 'sal-oe10':
        train_folds = SALZINNES_train_synthetic_overexposure10
        test_folds = SALZINNES_test_synthetic_overexposure10
    elif dbname == 'sal-oe0.4':
        train_folds = SALZINNES_train_synthetic_overexposure0_4
        test_folds = SALZINNES_test_synthetic_overexposure0_4
    elif dbname == 'sal-blur30x30':
        train_folds = SALZINNES_train_synthetic_blur30x30
        test_folds = SALZINNES_test_synthetic_blur30x30
    elif dbname == 'voy':
        train_folds = [val for sublist in DIBCO for val in sublist]
        test_folds = VOYNICH_test
    elif dbname == 'voy-ic':
        train_folds = [val for sublist in DIBCO_synthetic_inv_col for val in sublist]
        test_folds = VOYNICH_test_synthetic_inv_col
    elif dbname == 'bdi':
        train_folds = BDI_train
        test_folds = BDI_test
    elif dbname == 'bdi-ic':
        train_folds = BDI_train_synthetic_inv_col
        test_folds = BDI_test_synthetic_inv_col
    elif dbname == 'all':
        test_folds = [DIBCO[5], DIBCO[6]]
        test_folds.append(PALM_test[0])
        test_folds.append(PALM_test[1])
        test_folds.append(PHI_test)
        test_folds.append(EINSIELDELN_test)
        test_folds.append(SALZINNES_test)

        DIBCO.pop(6)
        DIBCO.pop(5)
        train_folds = [[val for sublist in DIBCO for val in sublist]]
        train_folds.append(PALM_train[0])
        train_folds.append(PALM_train[1])
        train_folds.append(PHI_train)
        train_folds.append(EINSIELDELN_train)
        train_folds.append(SALZINNES_train)

        test_folds = [val for sublist in test_folds for val in sublist]  # transform to flat lists
        train_folds = [val for sublist in train_folds for val in sublist]
    elif dbname == 'all-ic':
        test_folds = [DIBCO_synthetic_inv_col[5], DIBCO_synthetic_inv_col[6]]
        test_folds.append(PALM_test_synthetic_inv_col[0])
        test_folds.append(PALM_test_synthetic_inv_col[1])
        test_folds.append(PHI_test_synthetic_inv_col)
        test_folds.append(EINSIELDELN_test_synthetic_inv_col)
        test_folds.append(SALZINNES_test_synthetic_inv_col)

        DIBCO_synthetic_inv_col.pop(6)
        DIBCO_synthetic_inv_col.pop(5)
        train_folds = [[val for sublist in DIBCO_synthetic_inv_col for val in sublist]]
        train_folds.append(PALM_train_synthetic_inv_col[0])
        train_folds.append(PALM_train_synthetic_inv_col[1])
        train_folds.append(PHI_train_synthetic_inv_col)
        train_folds.append(EINSIELDELN_train_synthetic_inv_col)
        train_folds.append(SALZINNES_train_synthetic_inv_col)

        test_folds = [val for sublist in test_folds for val in sublist]  # transform to flat lists
        train_folds = [val for sublist in train_folds for val in sublist]
    else:
        raise Exception('Unknown database name')

    return train_folds, test_folds

