import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
import sys
import h5py


path = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/Decathlon/Brain/ProcessingOutputs/SingleFiles/data.h5'

hf = h5py.File(path, 'r')

print(hf.keys())
x_train = np.array(hf.get('x_train'))
y_train = np.array(hf.get('y_train'))

for i in range(x_train.shape[0]):

    if random.uniform(0, 1) > 0.9:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(x_train[i,:,:,0])

        plt.subplot(122)
        plt.imshow(y_train[i,:,:,0])
        plt.show()

a = 1