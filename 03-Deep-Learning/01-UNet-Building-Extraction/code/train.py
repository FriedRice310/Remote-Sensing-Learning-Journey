import torch
from utils import load_and_preprocess_data
from dataset import create_datasets
from model import UNet, train_model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

def main(image_file, mask_file):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    print("加载数据...")
    image, mask = load_and_preprocess_data(image_file, mask_file)
    
    print("创建数据集...")
    train_dataset, val_dataset = create_datasets(
        image, mask, 
        patch_size=256, 
        stride=128,
        train_ratio=0.8
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print("初始化模型...")
    model = UNet(n_channels=3, n_classes=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    print("开始训练...")
    train_losses, val_losses, val_ious = train_model(
        train_loader, val_loader, model, device, num_epochs=50
    )
    
    torch.save(model.state_dict(),r'Remote-Sensing-Learning-Journey\03-Deep-Learning\01-UNet-Building-Extraction\model\unet_model.pth')

    plot_training_curves(train_losses, val_losses, val_ious)

def plot_training_curves(train_losses, val_losses, val_ious):
    """绘制训练过程曲线"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练和验证损失')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_ious, label='验证IoU', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('验证集IoU')
    
    plt.tight_layout()
    plt.savefig(r'Remote-Sensing-Learning-Journey\03-Deep-Learning\01-UNet-Building-Extraction\out\training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    image_file = r'Remote-Sensing-Learning-Journey\03-Deep-Learning\01-UNet-Building-Extraction\dataset\images\1d4fbe33f3_F1BE1D4184INSPIRE-ortho.tif'
    mask_file = r'Remote-Sensing-Learning-Journey\03-Deep-Learning\01-UNet-Building-Extraction\dataset\labels\1d4fbe33f3_F1BE1D4184INSPIRE-label.png'
    main(image_file, mask_file)