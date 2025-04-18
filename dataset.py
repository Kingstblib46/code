# dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import common # 导入共享变量
import one_hot # 导入独热编码函数

class CaptchaDataset(Dataset):
    def __init__(self, root_dir):
        super(CaptchaDataset, self).__init__()
        self.root_dir = root_dir
        # 获取目录下所有 png 图片文件
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        if not self.image_paths:
             raise FileNotFoundError(f"No PNG images found in directory: {root_dir}")

        # 定义图像预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((common.image_height, common.image_width)), # 调整大小
            transforms.Grayscale(num_output_channels=1), # 转换为灰度图
            transforms.ToTensor(), # 转换为 Tensor，并将像素值缩放到 [0, 1]
            # transforms.Normalize((0.5,), (0.5,)) # 可选：归一化到 [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # 从文件名获取标签 (假设文件名为 label_timestamp_index.png)
        try:
            image_name = os.path.basename(image_path)
            label_text = image_name.split('_')[0]
            if len(label_text) != common.captcha_size:
                print(f"Warning: Skipping image with incorrect label length: {image_path}")
                # 如果标签长度不对，可以尝试加载下一个样本
                return self.__getitem__((index + 1) % len(self))
        except IndexError:
             print(f"Warning: Could not extract label from filename: {image_path}. Skipping.")
             return self.__getitem__((index + 1) % len(self))


        # 加载图片
        try:
            img = Image.open(image_path)
            # 应用转换
            img_tensor = self.transform(img)
        except Exception as e:
            print(f"Error loading or transforming image {image_path}: {e}. Skipping.")
            return self.__getitem__((index + 1) % len(self))


        # 将标签文本转换为独热编码向量
        try:
            label_vec = one_hot.text_to_vec(label_text)
             # 将独热编码向量扁平化以匹配 MultiLabelSoftMarginLoss 的期望输入
            label_vec_flat = label_vec.view(1, -1).squeeze() # Shape: (captcha_size * num_classes)
        except ValueError as e:
            print(f"Error encoding label '{label_text}' from {image_path}: {e}. Skipping.")
            return self.__getitem__((index + 1) % len(self))


        return img_tensor, label_vec_flat # 返回图像张量和扁平化的标签向量

# 测试数据加载 (可选)
if __name__ == '__main__':
    # 确保你已经生成了训练数据到 ./datasets/train
    train_dataset_path = './datasets/train'
    if not os.path.exists(train_dataset_path) or not os.listdir(train_dataset_path):
         print(f"Error: Training dataset directory '{train_dataset_path}' is empty or does not exist.")
         print("Please run generate_images.py first to create dataset.")
    else:
        print(f"Attempting to load data from: {train_dataset_path}")
        try:
            train_dataset = CaptchaDataset(root_dir=train_dataset_path)
            print(f"Dataset size: {len(train_dataset)}")

            # 创建 DataLoader
            # 利用你的 4090 资源可以增大 batch_size, e.g., 128, 256 or 512
            batch_size = 64
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers 加速数据加载

            # 获取一个批次的数据进行检查
            img_batch, label_batch = next(iter(train_dataloader))

            print(f"Image batch shape: {img_batch.shape}") # 应为 (batch_size, 1, height, width)
            print(f"Label batch shape: {label_batch.shape}") # 应为 (batch_size, captcha_size * num_classes)

            # 可视化一个样本 (需要 matplotlib)
            import matplotlib.pyplot as plt
            idx_to_show = 0
            plt.imshow(img_batch[idx_to_show].squeeze(), cmap='gray')
            # 解码标签需要 reshape 回 (captcha_size, num_classes)
            label_reshaped = label_batch[idx_to_show].view(common.captcha_size, common.num_classes)
            decoded_label = one_hot.vec_to_text(label_reshaped)
            plt.title(f"Label: {decoded_label}")
            plt.show()

            print("Dataset loading test successful.")

        except Exception as e:
             print(f"An error occurred during dataset loading test: {e}")