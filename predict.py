# predict.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm # 进度条
import random
import matplotlib.pyplot as plt
from PIL import Image

import common # 共享变量
from model import MyModel # 模型定义
from dataset import CaptchaDataset # 数据集类 (主要用于加载测试数据)
import one_hot # 编码函数

# --- 预测参数 ---
test_dataset_path = './datasets/test'
model_load_path = './model.pth'
# 预测时 batch_size 可以设置得更大，只要显存允许
batch_size = 128 # 可以在 4090 上设置更大, e.g., 256, 512, 1024
num_workers = 4
num_samples_to_show = 10 # 显示多少个预测样本
# --- 参数设置结束 ---

def predict():
    # 1. 加载测试数据
    print(f"Loading testing data from {test_dataset_path}...")
    try:
        test_dataset = CaptchaDataset(root_dir=test_dataset_path)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if common.device == 'cuda' else False)
        print(f"Test data loaded. Dataset size: {len(test_dataset)}, Batches: {len(test_dataloader)}")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        print("Please ensure you have run generate_images.py and the test dataset exists.")
        return

    # 2. 加载训练好的模型
    print(f"Loading trained model from {model_load_path}...")
    try:
        model = MyModel().to(common.device)
        model.load_state_dict(torch.load(model_load_path, map_location=common.device)) # 加载模型参数
        model.eval() # 设置为评估模式 (关闭 Dropout 等)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_load_path}. Please train the model first by running train.py.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. 进行预测和评估
    correct = 0
    total = 0
    all_results = [] # 存储 (image_path, true_label, predicted_label)

    print("Starting prediction...")
    with torch.no_grad(): # 预测时不需要计算梯度
        progress_bar = tqdm(test_dataloader, desc="Predicting")
        for imgs, targets in progress_bar:
            imgs = imgs.to(common.device)       # (B, 1, H, W)
            targets = targets.to(common.device) # (B, captcha_size * num_classes)

            outputs = model(imgs) # (B, captcha_size * num_classes)

            # Reshape for decoding
            outputs_reshaped = outputs.view(-1, common.captcha_size, common.num_classes)
            targets_reshaped = targets.view(-1, common.captcha_size, common.num_classes)

            predict_labels = one_hot.vec_to_text(outputs_reshaped) # List of strings
            true_labels = one_hot.vec_to_text(targets_reshaped)    # List of strings

            current_batch_size = len(true_labels)
            for i in range(current_batch_size):
                is_correct = (predict_labels[i] == true_labels[i])
                if is_correct:
                    correct += 1
                # 找到对应的图片路径 (这比较低效，更好的方法是在 Dataset 返回路径)
                # 为了简化，我们这里不直接关联路径，只记录结果
                all_results.append({
                    "true": true_labels[i],
                    "pred": predict_labels[i],
                    "correct": is_correct
                    # 'path': test_dataset.image_paths[total + i] # 如果需要路径
                })

            total += current_batch_size

            # 更新进度条显示当前准确率
            accuracy = (correct / total) * 100 if total > 0 else 0.0
            progress_bar.set_postfix({'acc': f'{accuracy:.2f}%'})

    # 4. 输出总体准确率
    final_accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"\nPrediction finished.")
    print(f"Total tested: {total}")
    print(f"Correctly predicted: {correct}")
    print(f"Accuracy: {final_accuracy:.2f}%")

    # 5. 可视化部分预测结果
    print(f"\nShowing {num_samples_to_show} sample predictions...")
    # 从测试集中随机选择一些图片进行展示
    random_indices = random.sample(range(len(test_dataset)), k=min(num_samples_to_show, len(test_dataset)))

    fig, axes = plt.subplots(nrows=(num_samples_to_show + 1) // 2, ncols=2, figsize=(10, num_samples_to_show * 1.5))
    axes = axes.flatten() # 将二维数组展平

    for i, idx in enumerate(random_indices):
        img_path = test_dataset.image_paths[idx]
        img_tensor, label_vec_flat = test_dataset[idx] # 获取预处理后的图像和标签

        # 模型预测单张图片
        img_tensor_batch = img_tensor.unsqueeze(0).to(common.device) # 增加 batch 维度
        output = model(img_tensor_batch)
        output_reshaped = output.view(1, common.captcha_size, common.num_classes)
        predicted_label = one_hot.vec_to_text(output_reshaped)

        # 解码真实标签
        label_reshaped = label_vec_flat.view(common.captcha_size, common.num_classes)
        true_label = one_hot.vec_to_text(label_reshaped)

        # 显示图片和标签
        img_display = Image.open(img_path) # 加载原始图片显示
        ax = axes[i]
        ax.imshow(img_display)
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}", color='green' if true_label == predicted_label else 'red')
        ax.axis('off')

    # 如果样本数量为奇数，隐藏最后一个空的子图
    if len(random_indices) % 2 != 0 and len(axes) > len(random_indices):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predict()