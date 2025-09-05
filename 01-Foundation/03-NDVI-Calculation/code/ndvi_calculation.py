import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

raster_path = r"Remote-Sensing-Learning-Journey\01-Foundation\Post-processing data\B{}_ref.tif"

NIR_band = gdal.Open(raster_path.format(5)).ReadAsArray().astype(np.float32)
RED_band = gdal.Open(raster_path.format(4)).ReadAsArray().astype(np.float32)

ADD_band = NIR_band + RED_band
ADD_band_mask = np.ma.masked_where(ADD_band == 0, ADD_band)
ndvi = (NIR_band - RED_band)/(ADD_band_mask)
ndvi.filled(0)

ndvi = np.clip(ndvi, -1, 1)

plt.imshow(ndvi, cmap= 'YlGn')
plt.colorbar(label='NDVI Value')
plt.axis('off')
plt.title('NDVI Image')
plt.show()
plt.imsave(r'Remote-Sensing-Learning-Journey\01-Foundation\03-NDVI-Calculation\rawresults\ndvi_image.png', ndvi, cmap= 'YlGn')