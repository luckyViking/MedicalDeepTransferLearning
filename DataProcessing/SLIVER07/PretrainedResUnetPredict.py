import DataProcessing.SLIVER07.PredictFunctions as functions

import os
import numpy as np
import pandas as pd
import random
import tqdm as tqdm
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model

from sklearn.metrics import jaccard_score

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

K.set_image_data_format('channels_last')
img_rows = 128
img_cols = 128
img_depth = 1

input_img = Input((128,128,1))
model = functions.get_unet(input_img, n_filters=32, dropout=0.3, batchnorm=True)

x_test = np.load(r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/ProcessedData/SingleFiles/x_test.npy')
y_test = np.load(r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/ProcessedData/SingleFiles/y_test.npy')



Weigths = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/Weights/ResUnetSliver_6.h5'
model.load_weights(Weigths)

learning_rate = 0.01
momentum = 0.7
nesterov = True
batch_size = 64
epochs = 100

sgd = SGD(lr=learning_rate, momentum = momentum, nesterov=nesterov)

model.compile(optimizer=sgd, loss=[jacard_coef_loss], metrics=[jacard_coef])

predictions = model.predict(x_test, batch_size = 1)

import random

scores = []

for prediction in range(predictions.shape[0]):

    pred = predictions[prediction, :, :, 0]
    gt = y_test[prediction, :, :, 0]

    pred_bin = np.where(pred < 0.2, 0, 1)
    gt_bin = np.where(gt < 0.2, 0, 1)

    score = jaccard_score(gt_bin.ravel(), pred_bin.ravel())
    scores.append(score)

    if random.uniform(0,1) > 0.8:



        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('JaccardScore = '+str(score))

        plt.subplot(131)
        plt.imshow(pred)

        plt.subplot(132)
        plt.imshow(gt)

        plt.subplot(133)
        plt.imshow(x_test[prediction, :, :, 0])
        plt.show()


Scores = np.asarray(scores)
np.save(r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/scores/scores_ResUnet_Sliver_weights_6.npy',Scores)
NonZeroScores = np.where(Scores == 0.0)
print(np.mean(NonZeroScores))

# TODO predict decathlon data, at least liver segmentationwise

a = 1





















