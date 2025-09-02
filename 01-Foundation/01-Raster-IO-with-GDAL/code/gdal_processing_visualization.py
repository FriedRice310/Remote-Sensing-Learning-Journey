from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

# 读取红绿蓝波段
raster_path = r"Remote-Sensing-Learning-Journey\01-Foundation\01-Raster-IO-with-GDAL\data\LC09_L1TP_121040_20250827_20250828_02_T1\LC09_L1TP_121040_20250827_20250828_02_T1_B{}.TIF"
red_band = gdal.Open(raster_path.format(4)).ReadAsArray()
red_band = red_band.astype(float)
green_band = gdal.Open(raster_path.format(3)).ReadAsArray()
green_band = green_band.astype(float)
blue_band = gdal.Open(raster_path.format(2)).ReadAsArray()
blue_band = blue_band.astype(float)

# 创建真彩色合成图像
def true_color_composite(red_band, green_band, blue_band):
    true_color_image = np.dstack((red_band, green_band, blue_band))

    # 归一化到0-1范围
    true_color_image = true_color_image / np.max(true_color_image)

    # 显示图像
    plt.imshow(true_color_image)
    plt.axis('off')
    plt.title('True Color Composite Image')
    plt.show()

    # 保存图像
#    plt.imsave(r'Remote-Sensing-Learning-Journey\01-Foundation\01-Raster-IO-with-GDAL\rawresults\true_color_image.png', true_color_image)

# 计算并显示NDVI
def ndvi_calculation(red_band, raster_path):
    # 读取近红外波段
    nir_band = gdal.Open(raster_path.format(5)).ReadAsArray()
    nir_band = nir_band.astype(float)
    # 计算NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.clip(ndvi, -1, 1)  # 限制NDVI值在-1到1之间

    # 显示NDVI图像
    plt.imshow(ndvi, cmap='RdYlGn', vmax=1, vmin=-1)
    plt.colorbar(label='NDVI Value')
    plt.axis('off')
    plt.title('NDVI Image')
    plt.show()  

    # 保存NDVI图像
#    plt.imsave(r'Remote-Sensing-Learning-Journey\01-Foundation\01-Raster-IO-with-GDAL\rawresults\ndvi_image.png', ndvi, cmap='RdYlGn', vmin=-1, vmax=1)

# 调试
ndvi_calculation(red_band, raster_path)
#true_color_composite(red_band, green_band, blue_band)