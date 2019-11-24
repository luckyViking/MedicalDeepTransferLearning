import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import skimage.transform

training_scans_dir = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/SLIVER07/training-scans/scan'
training_labels_dir = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/SLIVER07/training-labels/label'

OutputDir = r'DataProcessing/SLIVER07/ProcessedData'
ImgOutputPath = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/ProcessedData/Scan'
LabelOutputPath = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/ProcessedData/Label'
DataframeOutputPath = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Data/SLIVER/ProcessedData'


def CleanListFromRAW(list):
    CleanList = []
    for i in range(len(list)):
        CurrentListItem = list[i]
        if not (".mhd" in CurrentListItem):
            continue
        else:
            CleanList.append(CurrentListItem)
    return CleanList


def LoadMHD(filename):
    ITKimage = sitk.ReadImage(filename)

    ARRimg = sitk.GetArrayFromImage(ITKimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(ITKimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(ITKimage.GetSpacing())))

    return ARRimg, origin, spacing


def BuildDataframesAndSplitSlices(ImgFilenames, LabelFilenames, ImgDir, LabelDir, DataframeOutputPath, ImgOutputPath,
                                  LabelOutputPath, SaveSlices, Resample, SaveDF):
    IDs = np.arange(1, len(ImgFilenames) + 1, 1).tolist()

    Origins = []
    Spacings = []
    Slices = []

    SliceIDs = []
    SliceSpacings = []
    SliceOrigins = []
    SliceSlices = []
    ImgSliceFilenames = []
    LabelSliceFilenames = []
    SliceContainsLiver = []

    for i in range(len(ImgFilenames)):

        currentImg = ImgFilenames[i]
        currentLabel = LabelFilenames[i]

        ImgPath = os.path.join(ImgDir, currentImg)
        # print("Imagepath: " + ImgPath)
        ImgArr, ImgOrigin, ImgSpacing = LoadMHD(ImgPath)

        LabelPath = os.path.join(LabelDir, currentLabel)
        # print('Labelpath; ' + LabelPath)
        LabelArr, LabelOrigin, LabelSpacing = LoadMHD(LabelPath)

        if (ImgArr.shape[0] == LabelArr.shape[0]):
            numSlices = ImgArr.shape[0]
            Slices.append(ImgArr.shape[0])

        else:
            Slices.append(None)
            print("ERROR: Image and Label have different amount of slices")
            continue

        if np.array_equal(ImgOrigin, LabelOrigin):
            Origins.append(ImgOrigin)
        else:
            Origins.append(None)
            print("ERROR: Image and Label have different origin")

        if np.array_equal(ImgSpacing, LabelSpacing):
            Spacings.append(ImgSpacing)
        else:
            Spacings.append(None)
            print("ERROR: Image and Label have different spacing")

        for slice in range(numSlices):

            CurrentImgSlice = ImgArr[slice, :, :]
            CurrentLabelSlice = LabelArr[slice, :, :]

            if np.max(CurrentLabelSlice) == 0:
                SliceContainsLiver.append(0)
            else:
                SliceContainsLiver.append(1)

            # TODO Check if slice contains liver and write to df

            CurrentImgSliceName = "liver-orig" + str(IDs[i]) + "_slice" + str(slice + 1) + ".npy"
            CurrentLabelSliceName = "liver-seg" + str(IDs[i]) + "_slice" + str(slice + 1) + ".npy"

            # here you could also write to a different file format such as jpg, png...
            # ... or Resample...
            # ... and Augment...

            if Resample == True:
                # TODO after resizing, masks are not binary due to interpolation some values are between 0 and 1
                CurrentImgSlice = skimage.transform.resize(CurrentImgSlice, output_shape=[128, 128],
                                                           preserve_range=True)
                CurrentLabelSlice = skimage.transform.resize(CurrentLabelSlice, output_shape=[128, 128],
                                                             preserve_range=True)

            if SaveSlices == True:
                np.save(os.path.join(ImgOutputPath, CurrentImgSliceName), CurrentImgSlice)
                np.save(os.path.join(LabelOutputPath, CurrentLabelSliceName), CurrentLabelSlice)

            SliceIDs.append(IDs[i])
            ImgSliceFilenames.append(CurrentImgSliceName)
            LabelSliceFilenames.append(CurrentLabelSliceName)

            if (ImgArr.shape[0] == LabelArr.shape[0]):
                numSlices = ImgArr.shape[0]
                SliceSlices.append(ImgArr.shape[0])

            else:
                SliceSlices.append(None)
                print("ERROR: Image and Label have different amount of slices")
                continue

            if np.array_equal(ImgOrigin, LabelOrigin):
                SliceOrigins.append(ImgOrigin)
            else:
                SliceOrigins.append(None)
                print("ERROR: Image and Label have different origin")

            if np.array_equal(ImgSpacing, LabelSpacing):
                SliceSpacings.append(ImgSpacing)
            else:
                SliceSpacings.append(None)
                print("ERROR: Image and Label have different spacing")

    data = {'ID': IDs, 'Scan': ImgFilenames, 'Label': LabelFilenames, 'Spacing': Spacings,
            'Origin': Origins, 'Slices': Slices}

    dataSlices = {'ID': SliceIDs, 'Scan': ImgSliceFilenames, 'Label': LabelSliceFilenames, 'Spacing': SliceSpacings,
                  'Origin': SliceOrigins, 'Slices': SliceSlices, 'ContainsLiver': SliceContainsLiver}

    df = pd.DataFrame(data=data)
    slice_df = pd.DataFrame(data=dataSlices)

    if SaveDF == True:
        df.to_csv(os.path.join(DataframeOutputPath, 'train_df.csv'))
        slice_df.to_csv(os.path.join(DataframeOutputPath, 'sliced_train_df.csv'))

    return df, slice_df



training_labels_filenames = os.listdir(training_labels_dir)
training_labels_filenames.sort()

training_scans_filenames = os.listdir(training_scans_dir)
training_scans_filenames.sort()

training_labels_filenames = CleanListFromRAW(training_labels_filenames)
training_scans_filenames = CleanListFromRAW(training_scans_filenames)

train_df, sliced_train_df = BuildDataframesAndSplitSlices(training_scans_filenames, training_labels_filenames,
                                                          training_scans_dir, training_labels_dir,
                                                          DataframeOutputPath, ImgOutputPath, LabelOutputPath,
                                                          SaveSlices=True, Resample=True, SaveDF=True)


# TODO plot ratio between slices that contain liver and slices that dont