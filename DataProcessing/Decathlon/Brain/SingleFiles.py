import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
import sys

XTraiNDir = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/Decathlon/Brain/ProcessingOutputs/imagesTr'

XTrainContent = os.listdir(XTraiNDir)

x_train = []

for i in range(len(XTrainContent)):
    current = np.load(os.path.join(XTraiNDir, XTrainContent[i]))
    x_train.append(current)
    a = 1

SingleFilesOutput = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/Decathlon/Brain/ProcessingOutputs/SingleFiles'

np.save(os.path.join(SingleFilesOutput, 'x_train.npy'), x_train)




