import pandas as pd
import numpy as np
import os
from osgeo import gdal
from tqdm import tqdm

prediction_dir = ''
rgb_dir = ''

list_class = [0,1,2]
L = []
steps = 256
subsetsize = 256
filelist = os.listdir(prediction_dir)
for file in filelist:
    if os.path.splitext(file)[1] == '.tif':
        L.append(file)
frmt = 'GTiff'
driver = gdal.GetDriverByName(frmt)
for Id in tqdm(L):
    print(Id)
    labelds = gdal.Open(os.path.join(prediction_dir,Id))

    imX = labelds.RasterXSize
    imY = labelds.RasterYSize
    bands = labelds.RasterCount
    geo = labelds.GetGeoTransform()
    srs = labelds.GetProjection()
    dtype = labelds.GetRasterBand(1).DataType

    labelbuf1 = labelds.ReadAsArray()
    labelbuf2 = labelds.ReadAsArray()
    labelbuf3 = labelds.ReadAsArray()
    result_path = os.path.join(rgb_dir, '{}'.format(Id)).replace('\\','/')
    outlab = driver.Create(result_path, subsetsize, subsetsize, 3, gdal.GDT_Byte)
    outlab.SetGeoTransform(geo)
    outlab.SetProjection(srs)
    R = labelbuf1
    G = labelbuf2
    B = labelbuf3
    R[R==0] = 255
    R[R == 1] = 151
    R[R == 2] = 115
    R[R == 3] = 214
    R[R == 4] = 222
    G[G == 0] = 234
    G[G == 1] = 219
    G[G == 2] = 178
    G[G == 3] = 133
    G[G == 4] = 158
    B[B == 0] = 190
    B[B == 1] = 242
    B[B == 2] = 115
    B[B == 3] = 137
    B[B == 4] = 102
    outlab.GetRasterBand(1).WriteArray(R)
    outlab.GetRasterBand(2).WriteArray(G)
    outlab.GetRasterBand(3).WriteArray(B)
    labelds = None
