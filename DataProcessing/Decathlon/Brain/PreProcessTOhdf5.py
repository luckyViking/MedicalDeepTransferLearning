import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
import sys
import h5py

# examine data first

BraTSPath = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/Decathlon/Brain/Task01_BrainTumour/'

print(os.listdir(BraTSPath))

XTrainDir = os.path.join(BraTSPath, 'imagesTr')
YTrainDir = os.path.join(BraTSPath, 'labelsTr')

XTestDir = os.path.join(BraTSPath, 'imagesTs')

# PreProcess pre-coding
'''
"modality": { 
	 "0": "FLAIR", 
	 "1": "T1w", 
	 "2": "t1gd",
	 "3": "T2w"
 }
 
 "labels": { 
	 "0": "background", 
	 "1": "edema",
	 "2": "non-enhancing tumor",
	 "3": "enhancing tumour"
 } 
'''

FolderContentList = os.listdir(XTrainDir)

x_train = []
y_train = []
x_test = []
OutputPath = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/Decathlon/Brain/ProcessingOutputs/SingleFiles'
ImagesOutputPath = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/Decathlon/Brain/ProcessingOutputs/imagesTr'
LabelsOutputPath = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/Decathlon/Brain/ProcessingOutputs/labelsTr'


def ShrinkCase(x_train, y_train, fraction = 0.5, image_channel = 3, expand_dims = True):

    # picks random 50% (default) of the slices and only t2w images, to limit the amount of data

    if x_train.shape[2] == y_train.shape[2]:
        list = range(0, x_train.shape[2])
        sample = random.sample(list, int(x_train.shape[2]/2))
        sample.sort()

        x_train = x_train[:, :, sample, image_channel]
        y_train = y_train[:, :, sample]

        if expand_dims == True:
            x_train = np.expand_dims(x_train, 3)
            y_train = np.expand_dims(y_train, 3)

    else:
        print('x and y have different lengths/amount of slices...')

    return x_train, y_train

def SchrinkTestCase(x_test, fraction = 0.5, expand_dims = True):
    # TODO Test this function!!!
    list = range(0, x_test.shape[2])
    sample = random.sample(list, int(x_test.shape[2] / 2))
    sample.sort()
    x_test = x_test[:, :, sample]
    if expand_dims == True:
        x_train = np.expand_dims(x_test, 3)

    return x_test

PLOT = False

for i in tqdm(range(len(FolderContentList))):

    i_max = int(len(FolderContentList))

    CurrentFileName = FolderContentList[i]

    if not ("._" in CurrentFileName):

        # load the nii-volume
        x_train_i = nib.load(os.path.join(XTrainDir, CurrentFileName))
        x_train_i = np.array(x_train_i.dataobj)

        y_train_i = nib.load(os.path.join(YTrainDir, CurrentFileName))
        y_train_i = np.array(y_train_i.dataobj)

        x_train_i, y_train_i = ShrinkCase(x_train_i, y_train_i, fraction=0.5, image_channel=3)

        for ii in range(x_train_i.shape[2]):

            x_train.append(x_train_i[:, :, ii, :])
            y_train.append(y_train_i[:, :, ii, :])

            if random.uniform(0, 1) > 0.75 and PLOT == True:
                fig = plt.figure()
                plt.subplot(121)
                plt.imshow(x_train_i[:, :, ii, 0])

                plt.subplot(122)
                plt.imshow(y_train_i[:, :, ii, 0])
                plt.show()
            a = 1

    if i == int(len(FolderContentList)/2):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        np.save(os.path.join(OutputPath, 'x_train1.npy'), x_train)
        np.save(os.path.join(OutputPath, 'y_train1.npy'), y_train)
        x_train = []
        y_train = []

    if i == i_max:
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        np.save(os.path.join(OutputPath, 'x_train2.npy'), x_train)
        np.save(os.path.join(OutputPath, 'y_train2.npy'), y_train)

    else:
        continue

#x_train = np.asarray(x_train)
#y_train = np.asarray(y_train)

#hdfOutPath = os.path.join(OutputPath, 'BraTS_t2.h5')

#hf = h5py.File('full_data.h5', 'w')
#hf.create_dataset('x_train', data=x_train)
#hf.create_dataset('y_train', data=y_train)
#hf.close()

#np.save(os.path.join(OutputPath, 'x_train.npy'), x_train)
#np.save(os.path.join(OutputPath, 'y_train.npy'), y_train)

a = 1

#NewFileName = CurrentFileName[:-7]
#np.save(os.path.join(ImagesOutputPath, NewFileName), x_train_i)
#np.save(os.path.join(LabelsOutputPath, NewFileName), y_train_i)

