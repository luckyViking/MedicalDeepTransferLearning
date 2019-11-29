from Models.ResUNet2D.ResUNet2D import *


WeightsPath = r'/home/luca/Desktop/Master/LfD/DeepTL_git/Models/ResUNet2D/Weights/ResUnetSliver_6.h5'

input_img = Input((128,128,1))
model = get_unet(input_img, n_filters=32, dropout=0.3, batchnorm=True)
model.summary()
