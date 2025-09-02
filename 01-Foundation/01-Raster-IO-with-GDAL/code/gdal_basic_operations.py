from osgeo import gdal

# 基本读取
raster_path = r"Remote-Sensing-Learning-Journey\01-Foundation\01-Raster-IO-with-GDAL\data\LC09_L1TP_121040_20250827_20250828_02_T1\LC09_L1TP_121040_20250827_20250828_02_T1_B1.TIF"
raster = gdal.Open(raster_path)
if raster is None:
    print("Failed to open the raster file.")

# 影像元数据

# 特定波段打印为数组
#print(raster.GetRasterBand(1).ReadAsArray())

# 波段数
#print(raster.RasterCount)

# 地理变换参数
'''(339285.0, 30.0, 0.0, 3310215.0, 0.0, -30.0) -> 
(左上角x坐标, 像素宽度, 旋转角度, 左上角y坐标, 旋转角度, 像素高度)'''
#print(raster.GetGeoTransform())

# CRS信息
#print(raster.GetProjection())