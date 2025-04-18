# common.py
import torch

# 验证码字符集 (所有可能出现的字符)
captcha_array = list("0123456789abcdefghijklmnopqrstuvwxyz")
# 验证码长度
captcha_size = 4
# 字符集大小
num_classes = len(captcha_array)

# 设备选择 (优先使用 CUDA GPU，如果可用的话)
# 你的 4090 会在这里被检测到并使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图片尺寸 (根据模型和预处理调整)
image_height = 60
image_width = 160

print(f"Using device: {device}")
print(f"Captcha characters: {''.join(captcha_array)}")
print(f"Captcha size: {captcha_size}")
print(f"Number of classes: {num_classes}")