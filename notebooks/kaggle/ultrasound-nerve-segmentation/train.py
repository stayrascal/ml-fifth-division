from __future__ import print_function
from optparse import OptionParser
from skimage.transform import resize
import sys, os, shutil, random
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import flip_axis, random_channel_shift
from keras_plus import LearningRateDecay
from u_model import get_unet, IMG_COLS as img_cols, IMG_ROWS as img_rows
from data import load_train_data, load_test_data, load_patient_num
from augmentation import random_zoom, elastic_transform, random_rotation
from utils import save_pickle, load_pickle, count_enum

_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')


def preprocess(imgs, to_rows=None, to_cols=None):
    if to_rows is None or to_cols is None:
        to_rows = img_rows
        to_cols = img_cols
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], to_rows, to_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = resize(imgs[i, 0], (to_rows, to_cols), preserve_range=True)
    return imgs_p


class Learner(object):
    
    suffix = ''
    res_dir = os.path.join(_dir, 'res' + suffix)
    best_weight_path = os.path.join(res_dir, 'unet.hdf5')
    test_mask_res = os.path.join(res_dir, 'imgs_mask_test.npy')
    test_mask_exist_res = os.path.join(res_dir, 'imgs_mask_exist_test.npy')
    meanstd_path = os.path.join(res_dir, 'meanstd.dump')
    valid_data_path = os.path.join(res_dir, 'valid.npy')
    tensorboard_dir = os.path.join(res_dir, 'tb')
    
    def __init__(self, model_func, validation_split):
        self.model_func = model_func
        self.validation_split = validation_split
        self.__iter_res_dir = os.path.join(self.res_dir, 'res_iter')
        self.__iter_res_file = os.path.join(self.__iter_res_dir, '{epoch:02d}-{val_loss:.4f}.unet.hdf5')
        
    def _dir_init(self):
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)
        #iter clean
        if os.path.exists(self.__iter_res_dir):
            shutil.rmtree(self.__iter_res_dir)
        os.mkdir(self.__iter_res_dir)
    
    def save_meanstd(self):
        data = [self.mean, self.std]
        save_pickle(self.meanstd_path, data)
        
    @classmethod
    def load_meanstd(cls):
        print ('Load meanstd from %s' % cls.meanstd_path)
        mean, std = load_pickle(cls.meanstd_path)
        return mean, std
    
    @classmethod
    def save_valid_idx(cls, idx):
        save_pickle(cls.valid_data_path, idx)
        
    @classmethod
    def load_valid_idx(cls):
        return load_pickle(cls.valid_data_path)
    
    def _init_mean_std(self, data):
        data = np.array(data).astype('float32')
        self.mean, self.std = np.mean(data), np.std(data)
        self.save_meanstd()
        return data
    
    def get_object_existance(self, mask_array):
        return np.array([int(np.sum(mask_array[i, 0]) > 0) for i in range(len(mask_array))])

    def standartize(self, array, to_float=False):
        if to_float:
            array = np.array(array).astype('float32')
        if self.mean is None or self.std is None:
            raise ValueError('No mean/std is initialised')
        
        array -= self.mean
        array /= self.std
        return array

    @classmethod
    def norm_mask(cls, mask_array):
        mask_array = mask_array.astype('float32')
        mask_array /= 255.0
        return mask_array

    @classmethod
    def shuffle_train(cls, data, mask):
        perm = np.random.permutation(len(data))
        data = data[perm]
        mask = mask[perm]
        return data, mask

    @classmethod
    def split_train_and_valid_by_patient(cls, data, mask, validation_split, shuffle=False):
        print('Shuffle & split...')
        patient_nums = load_patient_num()
        patient_dict = count_enum(patient_nums)
        pnum = len(patient_dict)
        val_num = int(pnum * validation_split)
        patients = patient_dict.keys()
        if shuffle:
            random.shuffle(patients)
        val_p, train_p = patients[:val_num], patients[val_num:]
        train_indexes = [i for i, c in enumerate(patient_nums) if c in set(train_p)]
        val_indexes = [i for i, c in enumerate(patient_nums) if c in set(val_p)]
        x_train, y_train = data[train_indexes], mask[train_indexes]
        x_valid, y_valid = data[val_indexes], mask[val_indexes]
        cls.save_valid_idx(val_indexes)
        print ('val patients:', len(x_valid), val_p)
        print ('train patients:', len(x_train), train_p)
        return (x_train, y_train), (x_valid, y_valid)

    @classmethod
    def split_train_and_valid(cls, data, mask, validation_split, shuffle=False):
        print('Shuffle & split...')
        if shuffle:
            data, mask = cls.shuffle_train(data, mask)
        # data: (5635, 1, 80, 112), mask: (5635, 1, 80, 112)
        split_at = int(len(data) * (1. - validation_split)) # 4058
        x_train, x_valid = (data[0:split_at], data[split_at:])
        # x_train, x_valid = (_slice_arrays(data, 0, split_at), _slice_arrays(data, split_at))
        y_train, y_valid = (mask[0:split_at], mask[split_at:])
        # y_train, y_valid = (_slice_arrays(mask, 0, split_at), _slice_arrays(mask, split_at))
        cls.save_valid_idx(range(len(data))[split_at:])
        return (x_train, y_train), (x_valid, y_valid)
        
    def test(self, model, batch_size=256):
        print('Loading and pre-processing test data...')
        imgs_test = load_test_data()
        imgs_test = preprocess(imgs_test)
        imgs_test = self.standartize(imgs_test, to_float=True)
        imgs_test = imgs_test.transpose((0, 2, 3, 1))
    
        print('Loading best saved weights...')
        model.load_weights(self.best_weight_path)
        print('Predicting masks on test data and saving...')
        imgs_mask_test = model.predict(imgs_test, batch_size=batch_size, verbose=1)
        
        np.save(self.test_mask_res, imgs_mask_test[0])
        np.save(self.test_mask_exist_res, imgs_mask_test[1])
        
    def __pretrain_model_load(self, model, pretrained_path):
        if pretrained_path is not None:
            if not os.path.exists(pretrained_path):
                raise ValueError('No such pre-trained path exists')
            model.load_weights(pretrained_path)
            
            
    def augmentation(self, X, Y):
        print('Augmentation model...')
        total = len(X)
        x_train, y_train = [], []
        
        for i in range(total):
            x, y = X[i], Y[i]
            #standart
            x_train.append(x)
            y_train.append(y)
        
#            for _ in range(1):
#                _x, _y = elastic_transform(x[0], y[0], 100, 20)
#                x_train.append(_x.reshape((1,) + _x.shape))
#                y_train.append(_y.reshape((1,) + _y.shape))
            #flip x
            x_train.append(flip_axis(x, 2))
            y_train.append(flip_axis(y, 2))
            #flip y
            x_train.append(flip_axis(x, 1))
            y_train.append(flip_axis(y, 1))
            #continue
            #zoom
            for _ in range(1):
                _x, _y = random_zoom(x, y, (0.9, 1.1))
                x_train.append(_x)
                y_train.append(_y)
            for _ in range(0):
                _x, _y = random_rotation(x, y, 5)
                x_train.append(_x)
                y_train.append(_y)
            #intentsity
            for _ in range(1):
                _x = random_channel_shift(x, 5.0)
                x_train.append(_x)
                y_train.append(y)
    
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train
        
    def fit(self, x_train, y_train, x_valid, y_valid, pretrained_path):
        print('Creating and compiling and fitting model...')
        print('Shape:', x_train.shape)
        #second output
        y_train_2 = self.get_object_existance(y_train) # (22540,)
        y_valid_2 = self.get_object_existance(y_valid) # (1127,)

        x_train = x_train.transpose((0, 2, 3, 1))
        y_train = y_train.transpose((0, 2, 3, 1))
        x_valid = x_valid.transpose((0, 2, 3, 1))
        y_valid = y_valid.transpose((0, 2, 3, 1))

        #load model
        optimizer = Adam(lr=0.0045)
        model = self.model_func(optimizer)

        #checkpoints
        model_checkpoint = ModelCheckpoint(self.__iter_res_file, monitor='val_loss')
        model_save_best = ModelCheckpoint(self.best_weight_path, monitor='val_loss', save_best_only=True)
        early_s = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        learning_rate_adapt = LearningRateDecay(0.9, every_n=2, verbose=1)
        self.__pretrain_model_load(model, pretrained_path)
        model.fit(
                   x_train, [y_train, y_train_2], 
                   validation_data=(x_valid, [y_valid, y_valid_2]),
                   batch_size=128, nb_epoch=50,
                   verbose=1, shuffle=True,
                   callbacks=[model_save_best, model_checkpoint, early_s]
                   ) 
        
        #augment
        return model

    def filter_noise_data(self, imgs, imgs_mask):
        test_data = self.get_object_existance(imgs_mask)
        indexs = np.where(test_data > 0)
        return imgs[indexs], imgs_mask[indexs]

    def train_and_predict(self, pretrained_path=None, split_random=True):
        self._dir_init()
        print('Loading and preprocessing and standarize train data...')
        imgs_train, imgs_mask_train = load_train_data() # (5635, 1, 420, 580) (5635, 1, 420, 580)

        imgs_train, imgs_mask_train = self.filter_noise_data(imgs_train, imgs_mask_train)  # (5635, 1, 420, 580) (5635, 1, 420, 580)
        print('Img Shape: ', imgs_train.shape)


        imgs_train = preprocess(imgs_train) # (5635, 1, 80, 112)

        imgs_mask_train = preprocess(imgs_mask_train) # (5635, 1, 80, 112)
        
        imgs_mask_train = self.norm_mask(imgs_mask_train) # (5635, 1, 80, 112)

        split_func = split_random and self.split_train_and_valid or self.split_train_and_valid_by_patient
        (x_train, y_train), (x_valid, y_valid) = split_func(imgs_train, imgs_mask_train,
                                                        validation_split=self.validation_split)
        self._init_mean_std(x_train)
        x_train = self.standartize(x_train, True) # (5635, 1, 80, 112)
        x_valid = self.standartize(x_valid, True)
        #augmentation
        x_train, y_train = self.augmentation(x_train, y_train) # (22540, 1, 80, 112)
        #fit
        model = self.fit(x_train, y_train, x_valid, y_valid, pretrained_path)
        #test
        self.test(model)


def main():
    parser = OptionParser()
    parser.add_option("-s", "--split_random", action='store', type='int', dest='split_random', default = 1)
    parser.add_option("-m", "--model_name", action='store', type='str', dest='model_name', default = 'u_model')
    #
    options, _ = parser.parse_args()
    split_random = options.split_random
    model_name = options.model_name
    if model_name is None:
        raise ValueError('model_name is not defined')
    #
    import imp
    model_ = imp.load_source('model_', model_name + '.py')
    model_func = model_.get_unet
    #
    lr = Learner(model_func, validation_split=0.2)
    lr.train_and_predict(pretrained_path=None, split_random=split_random)
    print ('Results in ', lr.res_dir)

if __name__ == '__main__':
    sys.exit(main())
    # - loss: -0.2651 - main_output_loss: -0.3495 - aux_output_loss: 0.1689 - main_ou
    # tput_dice_coef: 0.3495 - aux_output_acc: 0.9970 - val_loss: -0.3890 - val_main_output_loss: -0.3900 - val_aux_output_loss: 0.0020 -
    # val_main_output_dice_coef: 0.3900 - val_aux_output_acc: 1.0000

    #
    # 22540 / 22540[ == == == == == == == == == == == == == == ==] - 646
    # s - loss: -0.6282 - main_output_loss: -0.6284 - aux_output_loss: 3.5119e-04 - main
    # _output_dice_coef: 0.6284 - aux_output_acc: 1.0000 - val_loss: -0.6305 - val_main_output_loss: -0.6305 - val_aux_output_loss: 1.8125
    # e - 05 - val_main_output_dice_coef: 0.6305 - val_aux_output_acc: 1.0000
    # Epoch
    # 00032: early
    # stopping
    # Loading and pre - processing
    # test
    # data...
    # Loading
    # test
    # data
    # from / notebooks / notebooks / kaggle / ultrasound - nerve - segmentation / np_data / imgs_test.npy
    # Loading
    # best
    # saved
    # weights...
    # Predicting
    # masks
    # on
    # test
    # data and saving...
    # 5508 / 5508[ == == == == == == == == == == == == == == ==] - 86
    # s

    # 646
    # s - loss: -0.6211 - main_output_loss: -0.6213 - aux_output_loss: 4.9209e-04 - main
    # _output_dice_coef: 0.6213 - aux_output_acc: 1.0000 - val_loss: -0.5926 - val_main_output_loss: -0.5926 - val_aux_output_loss: 3.2229
    # e - 05 - val_main_output_dice_coef: 0.5926 - val_aux_output_acc: 1.0000
    # Epoch
    # 00023: early
    # stopping
    # Loading and pre - processing
    # test
    # data...
    # Loading
    # test
    # data
    # from / notebooks / notebooks / kaggle / ultrasound - nerve - segmentation / np_data / imgs_test.npy
    # Loading
    # best
    # saved
    # weights...
    # Predicting
    # masks
    # on
    # test
    # data and saving...
    # 5508 / 5508[ == == == == == == == == == == == == == == ==] - 86
    # s


