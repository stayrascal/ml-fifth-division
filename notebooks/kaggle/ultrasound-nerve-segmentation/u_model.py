import sys
from keras import layers
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from metric import np_dice_coef, np_dice_coef_loss

IMG_ROWS, IMG_COLS = 80, 112 

def _shortcut(_input, residual):
    stride_width = _input._keras_shape[2] / residual._keras_shape[2]
    stride_height = _input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == _input._keras_shape[1]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(kernel_size=(1, 1), 
                      filters=residual._keras_shape[1], 
                      strides=(stride_width, stride_height), 
                      padding="valid",
                      kernel_initializer="he_normal")(_input)

    return layers.concatenate([shortcut, residual], axis=3)


def inception_block(inputs, depth, splitted=False, activation='relu'):
    assert depth % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
    
    c1_1 = Conv2D(depth//4, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)

    c2_1 = Conv2D(depth//8*3, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    c2_1 = actv()(c2_1)
    if splitted:
        c2_2 = Conv2D(depth//2, (1, 3), padding='same', kernel_initializer='he_normal')(c2_1)
        c2_2 = BatchNormalization(axis=1)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Conv2D(depth//2, (3, 1), padding='same', kernel_initializer='he_normal')(c2_2)
    else:
        c2_3 = Conv2D(depth//2, (3, 3), padding='same', kernel_initializer='he_normal')(c2_1)
    
    c3_1 = Conv2D(depth//16, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    c3_1 = actv()(c3_1)
    if splitted:
        c3_2 = Conv2D(depth//8, (1, 5), padding='same', kernel_initializer='he_normal')(c3_1)
        c3_2 = BatchNormalization(axis=1)(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Conv2D(depth//8, (5, 1), padding='same', kernel_initializer='he_normal')(c3_2)
    else:
        c3_3 = Conv2D(depth//8, (5, 5), padding='same', kernel_initializer='he_normal')(c3_1)

    p4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(inputs)
    c4_2 = Conv2D(depth//8, (1, 1), padding='same', kernel_initializer='he_normal')(p4_1)

    res = layers.concatenate([c1_1, c2_3, c3_3, c4_2], axis=3)
    res = BatchNormalization(axis=1)(res)
    res = actv()(res)
    return res
    

def rblock(inputs, num, depth, scale=0.1):    
    residual = Conv2D(depth, (num, num), padding='same')(inputs)
    residual = BatchNormalization(axis=1)(residual)
    residual = Lambda(lambda x: x*scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res) 
    

def NConv2D(kernel_size, filters, strides=(1, 1), padding='same'):
    def f(_input):
        conv = Conv2D(kernel_size=kernel_size, filters=filters, strides=strides, padding=padding)(_input)
        norm = BatchNormalization(axis=1)(conv)
        return ELU()(norm)

    return f

def BNA(_input):
    inputs_norm = BatchNormalization(axis=1)(_input)
    return ELU()(inputs_norm)

def reduction_a(inputs, k=64, l=64, m=96, n=96):
    "35x35 -> 17x17"
    inputs_norm = BNA(inputs)
    pool1 = MaxPooling2D((3,3), strides=(2,2), border_mode='same')(inputs_norm)

    conv2 = Conv2D(kernel_size=(3, 3), filters=n, strides=(2,2), padding='same')(inputs_norm)
    
    conv3_1 = NConv2D(kernel_size=(1, 1), filters=k, strides=(1, 1), padding='same')(inputs_norm)
    conv3_2 = NConv2D(kernel_size=(3, 3), filters=l, strides=(1, 1), padding='same')(conv3_1)
    conv3_2 = Conv2D(kernel_size=(3, 3), filters=m, strides=(2,2), padding='same')(conv3_2)
    
    res = layers.concatenate([pool1, conv2, conv3_2], concat_axis=1)
    return res


def reduction_b(inputs):
    "17x17 -> 8x8"
    inputs_norm = BNA(inputs)
    pool1 = MaxPooling2D((3,3), strides=(2,2), border_mode='same')(inputs_norm)
    #
    conv2_1 = NConv2D(kernel_size=(1, 1), filters=64, strides=(1, 1), padding='same')(inputs_norm)
    conv2_2 = Conv2D(kernel_size=(3, 3), filters=96, strides=(2,2), padding='same')(conv2_1)
    #
    conv3_1 = NConv2D(kernel_size=(1, 1), filters=64, strides=(1, 1), padding='same')(inputs_norm)
    conv3_2 = Conv2D(kernel_size=(3, 3), filters=72, strides=(2,2), padding='same')(conv3_1)
    #
    conv4_1 = NConv2D(kernel_size=(1, 1), filters=64, strides=(1, 1), padding='same')(inputs_norm)
    conv4_2 = NConv2D(kernel_size=(3, 3), filters=72, strides=(1, 1), padding='same')(conv4_1)
    conv4_3 = Conv2D(kernel_size=(3, 3), filters=80,  strides=(2,2), padding='same')(conv4_2)
    #
    res = layers.concatenate([pool1, conv2_2, conv3_2, conv4_3], axis=1)
    return res
    
    


def get_unet_inception_2head(optimizer):
    splitted = True
    act = 'elu'
    
    inputs = Input((IMG_ROWS, IMG_COLS, 1), name='main_input') # 80 x 112 x 1

    conv1 = inception_block(inputs, 32, splitted=splitted, activation=act)  # 80 x 112 x 32
    pool1 = NConv2D(3, 32, strides=(2, 2), padding='same')(conv1) # 40 x 56 x 32
    pool1 = Dropout(0.5)(pool1)
    
    conv2 = inception_block(pool1, 64, splitted=splitted, activation=act)  # 40 x 56 x 64
    pool2 = NConv2D(3, 64, strides=(2, 2), padding='same')(conv2) # 20 x 28 x 64
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = inception_block(pool2, 128, splitted=splitted, activation=act)  # 20 x 28 x 128
    pool3 = NConv2D(3, 128, strides=(2, 2), padding='same')(conv3) # 10 x 14 x 128
    pool3 = Dropout(0.5)(pool3)
     
    conv4 = inception_block(pool3, 256, splitted=splitted, activation=act)  # 10 x 14 x 256
    pool4 = NConv2D(3, 256, strides=(2, 2), padding='same')(conv4) # 5 x 7 x 256
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = inception_block(pool4, 512, splitted=splitted, activation=act)  # 5 x 7 x 512
    conv5 = Dropout(0.5)(conv5)
    
    #
    pre = Conv2D(1, (1, 1), activation="sigmoid", kernel_initializer="he_normal")(conv5)
    pre = Flatten()(pre)
    aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre) 
    #
    
    after_conv4 = rblock(conv4, 1, 256) # (?, 10, 14, 512)
    up6 = layers.concatenate([UpSampling2D(size=(2, 2))(conv5), after_conv4], axis=3) # (?, 10, 14, 1024)
    conv6 = inception_block(up6, 256, splitted=splitted, activation=act)  # (?, 10, 14, 256)
    conv6 = Dropout(0.5)(conv6)
    
    after_conv3 = rblock(conv3, 1, 128) # 20 x 28 x 256
    up7 = layers.concatenate([UpSampling2D(size=(2, 2))(conv6), after_conv3], axis=3) # (?, 20, 28, 512)
    conv7 = inception_block(up7, 128, splitted=splitted, activation=act)  # (?, 20, 28, 128)
    conv7 = Dropout(0.5)(conv7)
    
    after_conv2 = rblock(conv2, 1, 64) # 40 x 56 x 128
    up8 = layers.concatenate([UpSampling2D(size=(2, 2))(conv7), after_conv2], axis=3) # (?, 40, 56, 256)
    conv8 = inception_block(up8, 64, splitted=splitted, activation=act)  # (?, 40, 56, 64)
    conv8 = Dropout(0.5)(conv8)
    
    after_conv1 = rblock(conv1, 1, 32) # 80 x 112 x 64 
    up9 = layers.concatenate([UpSampling2D(size=(2, 2))(conv8), after_conv1], axis=3) # (?, 80, 112, 128)
    conv9 = inception_block(up9, 32, splitted=splitted, activation=act)  # (?, 80, 112, 32)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='main_output', kernel_initializer='he_normal')(conv9) # (?, 80, 112, 1)

    model = Model(input=inputs, output=[conv10, aux_out])
    model.compile(optimizer=optimizer,
                  loss={'main_output': np_dice_coef_loss, 'aux_output': 'binary_crossentropy'},
                  metrics={'main_output': np_dice_coef, 'aux_output': 'acc'},
                  loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model


get_unet = get_unet_inception_2head

def main():
    from keras.optimizers import Adam, RMSprop, SGD
    from metric import dice_coef, dice_coef_loss
    import numpy as np
    img_rows = IMG_ROWS
    img_cols = IMG_COLS
    
    optimizer = RMSprop(lr=0.045, rho=0.9, epsilon=1.0)
    model = get_unet(Adam(lr=1e-5))
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])
    
    x = np.random.random((1, 1,img_rows,img_cols))
    res = model.predict(x, 1)
    print(res)
    #print('res', res[0].shape)
    print('params', model.count_params())
    print('layer num', len(model.layers))
    #


if __name__ == '__main__':
    sys.exit(main())

