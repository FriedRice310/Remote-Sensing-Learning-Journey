from osgeo import gdal
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from skimage.measure import label, regionprops
SUN_ELEVATION = None
raster_path = r"Remote-Sensing-Learning-Journey\01-Foundation\data\LC09_L1TP_121040_20250827_20250828_02_T1\LC09_L1TP_121040_20250827_20250828_02_T1_B{}.TIF"
MTL_path = r"Remote-Sensing-Learning-Journey\01-Foundation\data\LC09_L1TP_121040_20250827_20250828_02_T1\LC09_L1TP_121040_20250827_20250828_02_T1_MTL.xml"
tree = ET.parse(MTL_path)
mult_and_add_dict = {}
ESUN = {1: 1969, 2: 1840, 3: 1551, 4: 1044, 5: 225.7, 6: 82.06}

# 读取波段数据
red_band = gdal.Open(raster_path.format(4)).ReadAsArray().astype(np.float32)
nir_band = gdal.Open(raster_path.format(5)).ReadAsArray().astype(np.float32)
blue_band = gdal.Open(raster_path.format(2)).ReadAsArray().astype(np.float32)
green_band = gdal.Open(raster_path.format(3)).ReadAsArray().astype(np.float32)

# 基于中值滤波的纯净暗区域选取(替换未定义值，效果不优，会追踪云的区域)
def get_dark(band):
    band[band == 0] = 9999
    print(1)
    thresh_mult = 1.1
    min_area = 100
    size = 50
    top_n = 3
    band_med = ndimage.median_filter(band, size)
    min_val = np.min(band_med)
    thresh = min_val * thresh_mult
    binary_image = band_med < thresh
    labeled, num_features = ndimage.label(binary_image)
    regions = regionprops(labeled, intensity_image=band)
    valid_regions = []
    for reg in regions:
        if reg.area< min_area:
            continue
        region_data = {
            'centroid': reg.centroid,
            'area': reg.area,
            'mean_intensity': reg.mean_intensity,
            'median_intensity': np.median(band[reg.coords[:,0], reg.coords[:,1]]),
            'coords': reg.coords
        }
        valid_regions.append(region_data)
    valid_regions = sorted(valid_regions, key=lambda x: x['median_intensity'])
    return valid_regions[:top_n]

# 基于选框的Lmin(不是为毛啊)
def get_Lmin_by_box(band):
    plt.imshow(band, cmap='gray')
    plt.axis('on')
    box = plt.ginput(2)
    water_patch = band[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]
    Lmin = np.median(water_patch)
    return Lmin

# 读取Lmin(未完善，假设为0)
def get_Lmin(band):
    return 0.0

# 预提取
def extract_mult_and_add(MTL_path):
    global mult_and_add_dict
    global SUN_ELEVATION
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
    mult_and_add_dict = mult_and_add
    SUN_ELEVATION = float(tree.getroot().find('IMAGE_ATTRIBUTES').find('SUN_ELEVATION').text)


# 读取传感器的增益和偏移量
def get_mult_and_add(band_number):
    return mult_and_add_dict.get(band_number, (0.0, 0.0))

# 辐射定标并计算地表反射率
def get_reflectance(dn, band_number, Lmin):
    RADIANCE_MULT_BAND, RADIANCE_ADD_BAND = get_mult_and_add(band_number)
    radiance = RADIANCE_MULT_BAND * dn + RADIANCE_ADD_BAND
    d = 1.0
    theta_s = np.radians(90 - SUN_ELEVATION)
    reflectance = (np.pi * (radiance - Lmin) * d**2) / (ESUN[band_number] * np.cos(theta_s))
    reflectance = np.clip(reflectance, 0, 1)
    return reflectance

# 校正后的真彩色合成
def plot_true_color_image(red, green, blue):
    red_band_ref = get_reflectance(red, 4, Lmin= get_Lmin(red))
    green_band_ref = get_reflectance(green, 3, Lmin= get_Lmin(green))
    blue_band_ref = get_reflectance(blue, 2, Lmin= get_Lmin(blue))
    true_color_image = np.dstack((red_band_ref, green_band_ref, blue_band_ref))
    plt.imshow(true_color_image)
    plt.title('True Color Image after Atmospheric Correction')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    extract_mult_and_add(MTL_path)
    plot_true_color_image(red_band, green_band, blue_band)