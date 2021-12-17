import pandas as pd
import numpy as np

import os
from osgeo import gdal
import glob
import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import keras
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal


def getImageArr(im, height=256, width=256):
    scaler = StandardScaler()
    try:
        ds = gdal.Open(im)
        buf = ds.ReadAsArray(0, 0, width, height)
        #buf = np.rollaxis(buf, 0, 3)
        buf = scaler.fit_transform(buf.astype(np.float32).reshape(-1,1)).reshape(3,256,256)
        buf = np.rollaxis(buf, 0, 3)
    except Exception as e:
        print(im + e)
        buf = np.zeros((height, width, 3))

    return buf


def getLabArr(lab, nclass=5, height=256, width=256):
    try:
        ds = gdal.Open(lab)
        buf = ds.ReadAsArray(0, 0, width, height)
        buf = np.rollaxis(buf, 0, 3)
    except Exception as e:
        print(e)

    lab_mark = np.reshape(buf, (height , width, nclass))
    return lab_mark


def imageLabelGenerator(img_path, lab_path, batchsize=10, nclass=9, input_height=128, input_width=128):
    imgs = glob.glob(os.path.join(img_path, "*.tif"))
    imgs.sort()
    labs = glob.glob(os.path.join(lab_path, "*.tif"))
    labs.sort()
    assert len(imgs) == len(labs)
    for im, lab in zip(imgs, labs):
        assert (os.path.basename(im) == os.path.basename(lab))
    zipped = itertools.cycle(zip(imgs, labs))
    while True:
        X = []
        Y = []
        for _ in range(batchsize):
            im, lab = next(zipped)
            X.append(getImageArr(im))
            Y.append(getLabArr(lab))
        yield np.array(X), np.array(Y)


def unnet_model():
    nclass = 5
    input_height = 256
    input_width = 256
    nb = 3
    inputs = Input(shape=(input_height, input_width, nb))
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block1_conv1')(inputs)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block1_conv2')(conv1)
    #conv1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D((2, 2),strides=(2,2),name='block1_pool')(conv1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block2_conv1')(pool1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block2_conv2')(conv2)
    #conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)
    #pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block3_conv1')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block3_conv2')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)
    #pool3 = MaxPooling2D((2, 2))(conv3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block4_conv1')(pool3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block4_conv2')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)
    #pool4 = MaxPooling2D((2, 2))(conv4)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block5_conv1')(pool4)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block5_conv2')(conv5)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02),name='block5_conv3')(conv5)

    #conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    #conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    #drop5 = Dropout(0.5)(conv5)

    #up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    up6 =UpSampling2D(size=(2, 2))(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    #up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    #up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    #up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(nclass, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(nclass, 1, activation='softmax')(conv9)
    #rrm
    '''x_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=1)(conv10)
    x_1 = BatchNormalization()(x_1)
    x_1 = Activation('relu')(x_1)

    x_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(x_1)
    x_2 = BatchNormalization()(x_2)
    x_2 = Activation('relu')(x_2)

    x_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=4)(x_2)
    x_3 = BatchNormalization()(x_3)
    x_3 = Activation('relu')(x_3)

    x_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=8)(x_3)
    x_4 = BatchNormalization()(x_4)
    x_4 = Activation('relu')(x_4)

    x_5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=16)(x_4)
    x_5 = BatchNormalization()(x_5)
    x_5 = Activation('relu')(x_5)

    x_6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=32)(x_5)
    x_6 = BatchNormalization()(x_6)
    x_6 = Activation('relu')(x_6)'''

    #rrm binglian
    x_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=1)(conv10)
    x_1 = BatchNormalization()(x_1)
    x_1 = Activation('relu')(x_1)

    x_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(x_1)
    x_2 = BatchNormalization()(x_2)
    x_2 = Activation('relu')(x_2)

    x_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=4)(x_2)
    x_3 = BatchNormalization()(x_3)
    x_3 = Activation('relu')(x_3)

    x_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=8)(x_3)
    x_4 = BatchNormalization()(x_4)
    x_4 = Activation('relu')(x_4)

    x_0_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv10)
    x_0_1 = BatchNormalization()(x_0_1)
    x_0_1 = Activation('relu')(x_0_1)

    x_0_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_0_1)
    x_0_2 = BatchNormalization()(x_0_2)
    x_0_2 = Activation('relu')(x_0_2)

    x_0_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_0_2)
    x_0_3 = BatchNormalization()(x_0_3)
    x_0_3 = Activation('relu')(x_0_3)

    x_0_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_0_3)
    x_0_4 = BatchNormalization()(x_0_4)
    x_0_4 = Activation('relu')(x_0_4)




    x = Add()([x_1, x_2, x_3,x_4,x_0_1,x_0_2,x_0_3,x_0_4])

    x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = Add()([conv10, x])

    outputs = Conv2D(filters=5, kernel_size=(1, 1), activation='softmax')(x)
    conv10 = Reshape((input_height , input_width, nclass))(outputs)

    model = Model(inputs=inputs, outputs=conv10)
    #model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])#'categorical_crossentropy'
    return model

if __name__ == '__main__':
    cnn_img_dir = '/media/aircas/Elements SE/小论文/数据集/area1/reclasses/train_image'
    cnn_label_dir = '/media/aircas/Elements SE/小论文/数据集/area1/reclasses/train_label'
    val_img_dir = '/media/aircas/Elements SE/小论文/数据集/area1/reclasses/val_image'
    val_label_dir = '/media/aircas/Elements SE/小论文/数据集/area1/reclasses/val_label'
    out_model = '/media/aircas/Elements SE/小论文/结果/unet/模型/unet_rrm_bl_3.h5'
    label_path = '/home/aircas/zsy/litixunjian/litixunjian/数据/label'
    #weight = get_weight(label_path)
    weight = {0:0.25,1:3,2:1.5,3:4,4:10}
    generator = imageLabelGenerator(cnn_img_dir, cnn_label_dir, batchsize=5)
    val_generator = imageLabelGenerator(val_img_dir,val_label_dir,batchsize=5)
    model = unnet_model()
    model_path = '/media/aircas/Elements SE/小论文/结果/unet/模型/unet109.h5'
    model.load_weights(model_path,by_name=True,skip_mismatch=True)
    model.compile(optimizer=Adam(lr=1e-4,decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint(out_model, monitor='loss', verbose=1, save_best_only=True)
    history = model.fit_generator(generator,steps_per_epoch=106*5*2, epochs=300, callbacks=[model_checkpoint],validation_data=val_generator,validation_steps=34*5*2)#validation_data=val_generator,validation_steps=33,
    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='upper left')
    plt.show()
    fig.savefig('/media/aircas/Elements SE/小论文/结果/unet/评价指标/unet_rrm_bl_3.png')
