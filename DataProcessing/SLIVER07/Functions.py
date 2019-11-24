import os
import numpy as np
import pandas as pd
import random
import tqdm as tqdm
import matplotlib.pyplot as plt

def TrainDataLoader(TrainImgPath, TrainLabelPath, df):
    # open df, shuffle and add 70% of the data to train, the other 30% to val
    # load data and split into train and val

    # df = sliced_train_df
    # shuffled_df = df.sample(frac=1)
    # shuffled_df = shuffled_df.reset_index(drop=True)
    shuffled_df = df
    ImgList = shuffled_df['Scan'].tolist()
    LabelList = shuffled_df['Label'].tolist()

    TestImgList = ImgList[-300:]
    TestLabelList = LabelList[-300:]

    ImgList = ImgList[:-300]
    LabelList = LabelList[:-300]

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    for i in range(len(TestImgList)):
        CurrentImage = np.load(os.path.join(TrainImgPath, TestImgList[i]))
        CurrentImage = np.expand_dims(CurrentImage, 2)
        x_test.append(CurrentImage)

        CurrentLabel = np.load(os.path.join(TrainLabelPath, TestLabelList[i]))
        CurrentLabel = np.expand_dims(CurrentLabel, 2)
        y_test.append(CurrentLabel)

    for img in range(len(ImgList)):
        CurrentImage = np.load(os.path.join(TrainImgPath, ImgList[img]))
        CurrentImage = np.expand_dims(CurrentImage, 2)

        CurrentLabel = np.load(os.path.join(TrainLabelPath, LabelList[img]))
        CurrentLabel = np.expand_dims(CurrentLabel, 2)

        if random.uniform(0, 1) > 0.7:
            x_val.append(CurrentImage)
            y_val.append(CurrentLabel)

        else:
            x_train.append(CurrentImage)
            y_train.append(CurrentLabel)

    print("Dataset \nTotal: " + str(len(ImgList)))
    print('Test: ' + str(len(TestImgList)))
    print('Train:' + str(len(x_train)))
    print('Validate:' + str(len(x_val)))

    return x_train, y_train, x_val, y_val, x_test, y_test


def PlotRandom(x, y, HowMany):
    for i in range(HowMany):
        RandomInteger = random.randint(0, len(x))

        imgSlice = x[RandomInteger]
        LabelSlice = y[RandomInteger]

        fig = plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(imgSlice[:, :, 0], cmap="gray")
        plt.title("Scan")

        plt.subplot(1, 2, 2)
        plt.imshow(LabelSlice[:, :, 0], cmap='gray')
        plt.title("Mask")

        plt.show()


def ShuffleDF(df):
    shuffled_df = df.sample(frac=1)
    shuffled_df = shuffled_df.reset_index(drop=True)

    return shuffled_df