import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet
from utils import load_and_preprocess_data

plt.rcParams['font.sans-serif'] = ['SimHei']

def predict_and_visualize(model, image, device, patch_size=256, stride=128):
    """对整个大图进行预测并可视化结果"""
    
    # 加载最佳模型
    model.load_state_dict(torch.load(r'Remote-Sensing-Learning-Journey\03-Deep-Learning\01-UNet-Building-Extraction\model\best_unet_model.pth'))
    model.eval()
    
    height, width = image.shape[:2]
    full_prediction = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
    
    with torch.no_grad():
        # 滑动窗口预测
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # 提取patch
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # 预处理
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
                
                # 预测
                output = model(patch_tensor)
                pred = torch.softmax(output, dim=1)[0, 1].cpu().numpy()  # 建筑物概率
                
                # 累加到完整预测图
                full_prediction[y:y+patch_size, x:x+patch_size] += pred
                count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # 平均重叠区域
    full_prediction = full_prediction / count_map
    
    return full_prediction

def visualize_results(original_image, ground_truth, prediction, threshold=0.5):
    """可视化原始图像、真值和预测结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 原始图像
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 真值标注
    axes[0, 1].imshow(ground_truth, cmap='jet')
    axes[0, 1].set_title('真值标注')
    axes[0, 1].axis('off')
    
    # 预测概率图
    im = axes[1, 0].imshow(prediction, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('预测概率图')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 二值化预测结果
    binary_pred = (prediction > threshold).astype(np.uint8)
    axes[1, 1].imshow(binary_pred, cmap='gray')
    axes[1, 1].set_title(f'二值化预测 (阈值={threshold})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(r'Remote-Sensing-Learning-Journey\03-Deep-Learning\01-UNet-Building-Extraction\out\prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return binary_pred

def run_inference(image_file, mask_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = UNet(n_channels=3, n_classes=2).to(device)
    
    # 加载测试图像
    test_image, test_mask = load_and_preprocess_data(image_file, mask_file)
    
    # 进行预测
    print("进行推理...")
    prediction = predict_and_visualize(model, test_image, device)
    
    # 可视化结果
    binary_pred = visualize_results(test_image, test_mask, prediction, threshold=0.5)
    
image_file = r'Remote-Sensing-Learning-Journey\03-Deep-Learning\01-UNet-Building-Extraction\dataset\images\1d4fbe33f3_F1BE1D4184INSPIRE-ortho.tif'
mask_file = r'Remote-Sensing-Learning-Journey\03-Deep-Learning\01-UNet-Building-Extraction\dataset\labels\1d4fbe33f3_F1BE1D4184INSPIRE-label.png'
run_inference(image_file, mask_file)