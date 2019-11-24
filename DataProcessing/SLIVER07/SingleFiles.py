import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import skimage.transform
from DataProcessing.SLIVER07.Functions import *


training_scans_dir = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/ProcessedData/Scan'
training_labels_dir = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/ProcessedData/Label'

df_path = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/ProcessedData/sliced_train_df.csv'

train_df = pd.read_csv(df_path)
#train_df = ShuffleDF(train_df)
train_df.head(5)

x_train, y_train, x_val, y_val, x_test, y_test = TrainDataLoader(training_scans_dir, training_labels_dir, train_df)


'''
SingleFilesOutput = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/ProcessedData/SingleFiles'
np.save(os.path.join(SingleFilesOutput, 'x_train.npy'), x_train)
np.save(os.path.join(SingleFilesOutput, 'y_train.npy'), y_train)
np.save(os.path.join(SingleFilesOutput, 'x_val.npy'), x_val)
np.save(os.path.join(SingleFilesOutput, 'y_val.npy'), y_val)
np.save(os.path.join(SingleFilesOutput, 'x_test.npy'), x_test)
np.save(os.path.join(SingleFilesOutput, 'y_test.npy'), y_test)
'''
a = 1