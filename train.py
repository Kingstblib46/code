# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # 进度条

import common # 共享变量
from model import MyModel # 模型定义
from dataset import CaptchaDataset # 数据集类
import one_hot # 编码函数

# --- 训练参数 ---
train_dataset_path = './datasets/train'
model_save_path = './model.pth'
learning_rate = 0.001
# 利用你的 4090 资源可以增加 epochs, e.g., 20, 50, 100
epochs = 10
# 根据你的 4090 显存调整 batch_size, e.g., 128, 256, 512
batch_size = 64
# 数据加载进程数，可以根据 CPU 核心数调整, e.g., 4, 8
num_workers = 4
# --- 参数设置结束 ---

def train():
    # 1. 加载数据
    print(f"Loading training data from {train_dataset_path}...")
    try:
        train_dataset = CaptchaDataset(root_dir=train_dataset_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if common.device == 'cuda' else False) # pin_memory 加速 GPU 传输
        print(f"Data loaded. Dataset size: {len(train_dataset)}, Batches: {len(train_dataloader)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have run generate_images.py and the dataset exists.")
        return

    # 2. 初始化模型、损失函数、优化器
    print("Initializing model...")
    model = MyModel().to(common.device) # 将模型移到 GPU (如果可用)

    # 使用 MultiLabelSoftMarginLoss 作为损失函数
    # 它期望的输入: (N, C) N 是 Batch size, C 是总类别数 (captcha_size * num_classes)
    # 它期望的目标: (N, C) 与输入形状相同，包含 0 或 1
    loss_fn = nn.MultiLabelSoftMarginLoss().to(common.device)

    # 使用 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    # 3. 训练循环
    for epoch in range(epochs):
        model.train() # 设置为训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # 使用 tqdm 显示进度条
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")

        for step, (imgs, targets) in progress_bar:
            imgs = imgs.to(common.device)       # (B, 1, H, W)
            targets = targets.to(common.device) # (B, captcha_size * num_classes)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(imgs) # (B, captcha_size * num_classes)

            # 计算损失
            loss = loss_fn(outputs, targets)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

            # 计算准确率 (可选, 可能会稍微减慢训练速度)
            # 需要将输出和目标 reshape 成 (B, captcha_size, num_classes)
            outputs_reshaped = outputs.view(-1, common.captcha_size, common.num_classes)
            targets_reshaped = targets.view(-1, common.captcha_size, common.num_classes)

            predict_labels = one_hot.vec_to_text(outputs_reshaped) # List of strings
            true_labels = one_hot.vec_to_text(targets_reshaped)    # List of strings

            current_batch_size = len(true_labels)
            for i in range(current_batch_size):
                 if predict_labels[i] == true_labels[i]:
                     correct_predictions += 1
            total_predictions += current_batch_size
            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0


            # 更新进度条显示损失和准确率
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})


        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # 4. 保存模型
    print(f"Training finished. Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path) # 只保存模型参数

if __name__ == "__main__":
    train()