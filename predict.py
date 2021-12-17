import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from tqdm import tqdm
from osgeo import gdal
from keras.models import *
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

nclass = 5
imagedir = '/media/aircas/Elements SE/小论文/数据集/area1/reclasses/test_image'
model_path = '/media/aircas/Elements SE/小论文/结果/unet/模型/unet_rrm_bl_2.h5'
out_dir = '/media/aircas/Elements SE/小论文/结果/unet/prediction_unet_rrm_bl_2'

blocksize = 256

# 输出分类后的图像
L = []
filelist = os.listdir(imagedir)
for file in filelist:
    if os.path.splitext(file)[1] == '.tif':
        L.append(file)

#自定义的损失函数,通过字典进行指定
mod = load_model(model_path)

scaler = StandardScaler()

for Id in tqdm(L):
    inDs = gdal.Open(os.path.join(imagedir,Id))
    nb = inDs.RasterCount
    cols = inDs.RasterXSize
    rows = inDs.RasterYSize
    geo = inDs.GetGeoTransform()
    srs = inDs.GetProjection()
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(os.path.join(out_dir,Id), cols, rows, 1, gdal.GDT_Byte)
    outDs.SetGeoTransform(geo)
    outDs.SetProjection(srs)
    mask = np.zeros((nb, cols, rows), dtype=np.int8)
    X = inDs.ReadAsArray()
    X = scaler.fit_transform(X.astype(np.float32).reshape(-1, 1)).reshape(3, 256, 256)
    X = np.rollaxis(X, 0, 3)
    X = np.expand_dims(X, axis=0)
    pr = mod.predict(X)  # [0]
    pr = np.reshape(pr, (cols, rows, nclass))
    class_type = pr.argmax(axis=2)
    outDs.GetRasterBand(1).WriteArray(class_type)
    inDs = None
    outDs = None