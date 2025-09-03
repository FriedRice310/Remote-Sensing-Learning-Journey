from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

raster_path = r"Remote-Sensing-Learning-Journey\01-Foundation\data\LC09_L1TP_121040_20250827_20250828_02_T1\LC09_L1TP_121040_20250827_20250828_02_T1_B{}.TIF"
MTL_path = r"Remote-Sensing-Learning-Journey\01-Foundation\data\LC09_L1TP_121040_20250827_20250828_02_T1\LC09_L1TP_121040_20250827_20250828_02_T1_MTL.xml"
tree = ET.parse(MTL_path)

ESUN = {1: 1969, 2: 1840, 3: 1551, 4: 1044, 5: 225.7, 6: 82.06}

# 读取波段数据
red_band = gdal.Open(raster_path.format(4)).ReadAsArray().astype(np.float32)
nir_band = gdal.Open(raster_path.format(5)).ReadAsArray().astype(np.float32)
blue_band = gdal.Open(raster_path.format(2)).ReadAsArray().astype(np.float32)
green_band = gdal.Open(raster_path.format(3)).ReadAsArray().astype(np.float32)

# 读取Lmin(未完善，假设为0)
def get_Lmin(band_number):
    return 0.0

# 预提取
def extract_mult_and_add(MTL_path):
    mult_and_add = {}
    tree = ET.parse(MTL_path)
    root = tree.getroot()
    for elem in root.iter():
        if elem.tag == 'LEVEL1_RADIOMETRIC_RESCALING':
            for band_number in [2,3,4,5]:
                try:
                    mult = float(elem.find(f'RADIANCE_MULT_BAND_{band_number}').text)
                    add = float(elem.find(f'RADIANCE_ADD_BAND_{band_number}').text)
                    mult_and_add[band_number] = (mult, add)
                except AttributeError:
                    continue
    return mult_and_add

mult_and_add_dict = extract_mult_and_add(MTL_path)

# 读取太阳天顶角
SUN_ELEVATION = float(tree.getroot().find('IMAGE_ATTRIBUTES').find('SUN_ELEVATION').text)

# 读取传感器的增益和偏移量
def get_mult_and_add(band_number):
    return mult_and_add_dict.get(band_number, (0.0, 0.0))

# 辐射定标并计算地表反射率
def get_reflectance(dn, band_number):
    RADIANCE_MULT_BAND, RADIANCE_ADD_BAND = get_mult_and_add(band_number)
    radiance = RADIANCE_MULT_BAND * dn + RADIANCE_ADD_BAND
    d = 1.0
    theta_s = np.radians(90 - SUN_ELEVATION)
    reflectance = (np.pi * (radiance - get_Lmin(band_number)) * d**2) / (ESUN[band_number] * np.cos(theta_s))
    reflectance = np.clip(reflectance, 0, 1)
    return reflectance

# 灰度显示校正后的红波段
plt.imshow(get_reflectance(red_band, 4), cmap='gray')
plt.colorbar(label='Reflectance')
plt.axis('off')
plt.show()
plt.imsave(r'Remote-Sensing-Learning-Journey\01-Foundation\02-Atmospheric-Correction\rawresults\red_band_reflectance.png', get_reflectance(red_band, 4), cmap='gray')

