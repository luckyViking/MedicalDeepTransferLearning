import numpy as np
import matplotlib.pyplot as plt
import os

# loading preprocessed data
# imagevolumes were split into slices and then into train, val and test (images(x) and labels(y))
# the slices then were stacked into single numpy arrays, to make them easier to load

SingleFiles = r'/home/luca/Desktop/Master/LfD/Data/Decathlon/SingleFiles'

x_train = np.load(os.path.join(SingleFiles, 'x_train.npy'))
y_train = np.load(os.path.join(SingleFiles, 'y_train.npy'))

x_test = np.load(os.path.join(SingleFiles, 'x_test.npy'))
y_test = np.load(os.path.join(SingleFiles, 'y_test.npy'))


a = 1