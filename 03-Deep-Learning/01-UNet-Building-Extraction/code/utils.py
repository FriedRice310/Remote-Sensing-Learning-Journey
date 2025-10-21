from osgeo import gdal
import numpy as np
import torch

def choose_class(mask, color = (75, 25, 230)):
    x0 = (mask[0] == color[0])
    x1 = (mask[1] == color[1])
    x2 = (mask[2] == color[2])
    y = (x0 | x1 | x2).astype(np.int64)
    return y

# 加载数据
def load_and_preprocess_data(image_file, mask_file):
    image = gdal.Open(image_file).ReadAsArray()
    mask = gdal.Open(mask_file).ReadAsArray()
    image = np.transpose(image[:3], (1, 2, 0)).astype(np.float32)/255.0
    mask = choose_class(mask, color = (75, 25, 230))
    return image, mask

# 数据增强
class Transform:
    def __call__(self, image, mask):
        # 随机水平翻转
        if np.random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        # 随机垂直翻转
        if np.random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        # 随机旋转90度
        if np.random.random() > 0.5:
            k = np.random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        
        return image, mask
    
def calculate_iou(output, target):
    # 将输出转换为预测类别
    pred = torch.argmax(output, dim=1)
    
    # 只计算前景（建筑物）的IoU
    intersection = ((pred == 1) & (target == 1)).float().sum()
    union = ((pred == 1) | (target == 1)).float().sum()
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()