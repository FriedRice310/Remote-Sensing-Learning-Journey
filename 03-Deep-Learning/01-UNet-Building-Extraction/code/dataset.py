import torch
from torch.utils.data import Dataset, random_split
from utils import Transform

# 滑动窗口加数据集构造
class LargeImageDataset(Dataset):
    def __init__(self, image, mask, patch_size=256, stride=128, transform=None):
        self.image = image
        self.mask = mask
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        self.height, self.width = image.shape[:2]
        self.patches_coords = []

        for x in range(0, self.height - self.patch_size + 1, stride):
            for y in range(0, self.width - self.patch_size + 1, stride):
                self.patches_coords.append((x, y))

    def __len__(self):
        return len(self.patches_coords)
    
    def __getitem__(self, index):
        x, y = self.patches_coords[index]
        image_patch = self.image[x:x+self.patch_size, y:y+self.patch_size]
        mask_patch = self.mask[x:x+self.patch_size, y:y+self.patch_size]
        
        # 确保 image_patch 的通道维度在第 0 维
        image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).float()
        
        # 确保 mask_patch 是 2D 的 (H, W)
        if mask_patch.ndim == 3:  # 如果 mask 是 3D 的
            mask_patch = mask_patch[:, :, 0]  # 取第一个通道（假设是单通道掩码）
        mask_patch = torch.from_numpy(mask_patch).long()
        
        if self.transform:
            image_patch, mask_patch = self.transform(image_patch, mask_patch)
            
        return image_patch, mask_patch

# 数据集创建
def create_datasets(image, mask, patch_size=256, stride=128, train_ratio=0.8):
    full_dataset = LargeImageDataset(
        image,
        mask,
        patch_size=256,
        stride=128,
        transform=None)

    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    return train_dataset, val_dataset