import joblib
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 

plt.rcParams['font.sans-serif'] = ['SimHei']
band_path = r"Remote-Sensing-Learning-Journey\02-Machine-Learning\01-Image-Classification-SVM\useMyDataset\dataset\images\data.tif"
color = r"Remote-Sensing-Learning-Journey\temporary\data\color.tif"
svm = joblib.load(r'Remote-Sensing-Learning-Journey\02-Machine-Learning\01-Image-Classification-SVM\useMyDataset\model\svm.pkl')
X = gdal.Open(band_path).ReadAsArray().astype(np.float32)
X_transpose = np.transpose(X,(1, 2, 0))
colors = ['blue', 'darkgreen', 'lightgreen', 'yellow', 'orange', 'darkblue']
class_names = ['水域', '森林', '农田', '房屋', '裸地', '湿地']
cmp = ListedColormap(colors)


def crop_array(X):

    colorimg = np.transpose(gdal.Open(color).ReadAsArray().astype(np.float32), (1, 2, 0))
    plt.imshow(colorimg)
    plt.axis('on')
    points = plt.ginput(2)
    plt.close()

    x1, y1 = int(min(points[0][0], points[1][0])), int(min(points[0][1], points[1][1]))
    x2, y2 = int(max(points[0][0], points[1][0])), int(max(points[0][1], points[1][1]))    
    cropped_array = X[y1:y2, x1:x2]

    plt.imshow(colorimg[y1:y2, x1:x2])
    plt.axis('on')
    plt.show()
    plt.close()

    return cropped_array

def pi(X_transpose, model):
    scaler = StandardScaler()
    height, width, num_bands = X_transpose.shape
    total_pixels = height * width

    all_pixels = X_transpose.reshape(-1, num_bands)

    chunk_size = 10000

    predictions = np.zeros(total_pixels, dtype=np.uint8)

    for start_idx in tqdm(range(0, total_pixels, chunk_size), desc="分类进度"):

        end_idx = min(start_idx + chunk_size, total_pixels)
        
        chunk_data = all_pixels[start_idx:end_idx, :]

        X = scaler.fit_transform(chunk_data)

        chunk_predictions = model.predict(X)
        
        predictions[start_idx:end_idx] = chunk_predictions

    classified_image = predictions.reshape(height, width)
    return classified_image

def save(X, original_path, save_path):
    original = gdal.Open(original_path)
    driver = gdal.GetDriverByName('GTiff')
    out_tif = driver.Create(
        save_path,
        original.RasterXSize,
        original.RasterYSize,
        1,
        gdal.GDT_Float32
    )
    out_tif.SetGeoTransform(original.GetGeoTransform())
    out_tif.SetProjection(original.GetProjection())
    out_tif.GetRasterBand(1).WriteArray(X)
    out_tif.FlushCache()
    out_tif = None

cropped_array = crop_array(X_transpose)
# o_shape = (cropped_array.shape[0], cropped_array.shape[1])
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(cropped_array.reshape(-1, 7))

# img = svm.predict(X_scaled).reshape(np.array(o_shape))

img = pi(cropped_array, svm)
plt.imshow(img, cmap=cmp)

legend_elements = [Patch(facecolor=colors[i], label=class_names[i]) 
                  for i in range(len(class_names))]

plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()
# save(img, color, r"Remote-Sensing-Learning-Journey\02-Machine-Learning\01-Image-Classification-SVM\useMyDataset\rawresults\lab.tif")