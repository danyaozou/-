import numpy as np
import os
from osgeo import gdal
from tqdm import tqdm

prediction_dir = ''
rgb_dir = ''

RGB = [0,1,2]
L = []
filelist = os.listdir(prediction_dir)
for file in filelist:
    if os.path.splitext(file)[1] == '.tif':
        L.append(file)

frmt = 'GTiff'
driver = gdal.GetDriverByName(frmt)
for Id in tqdm(L):
    print(Id)
    labelds = gdal.Open(os.path.join(prediction_dir, Id))

    imX = labelds.RasterXSize
    imY = labelds.RasterYSize
    bands = labelds.RasterCount
    geo = labelds.GetGeoTransform()
    srs = labelds.GetProjection()
    dtype = labelds.GetRasterBand(1).DataType

    labelbuf = labelds.ReadAsArray()
    result_path = os.path.join(rgb_dir,{}.format(Id)).replace('\\','/')#windows 路径反斜杠更改
    outlab = driver.Create(result_path,imX,imY,3,gdal.GDT_Byte)
    outlab.SetGeoTransform(geo)
    outlab.SetProjection(srs)
    R = labelbuf
    G = labelbuf
    B = labelbuf
    R[R ==0] =
    R[R ==1] =
    R[R ==] =
    R[R ==] =
    R[R ==] =
    G[G ==] =
    G[G ==] =
    G[G ==] =
    G[G ==] =
    B[B ==] =

    for a in RGB:
        outlab.GetRasterBand(a+1)