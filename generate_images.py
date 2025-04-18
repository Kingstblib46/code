# generate_images.py
import os
import random
import time
from captcha.image import ImageCaptcha
import common # 导入共享变量

# --- 参数设置 ---
# 你可以增加这里的数量来生成更多数据
num_train_images = 50000  # 训练图片数量 (利用你的 4090 资源可以生成更多, 比如 50000 或 100000)
num_test_images = 10000   # 测试图片数量 (例如 10000)
output_train_dir = "./datasets/train"
output_test_dir = "./datasets/test"
# --- 参数设置结束 ---

def generate_captcha_images(num_images, output_dir):
    """生成指定数量的验证码图片并保存到目录"""
    image_gen = ImageCaptcha(width=common.image_width, height=common.image_height)
    os.makedirs(output_dir, exist_ok=True) # 确保目录存在
    print(f"Generating {num_images} images to {output_dir}...")

    # 使用 tqdm 显示进度
    from tqdm import tqdm
    for i in tqdm(range(num_images), desc=f"Generating to {os.path.basename(output_dir)}"):
        # 随机生成验证码文本
        image_val = "".join(random.sample(common.captcha_array, common.captcha_size))
        # 生成包含时间戳的文件名，避免重名
        image_name = f"{image_val}_{int(time.time() * 1000)}_{i}.png"
        image_path = os.path.join(output_dir, image_name)

        try:
            # 生成并写入图片文件
            image_gen.write(image_val, image_path)
        except Exception as e:
            print(f"Error writing file {image_path}: {e}")

if __name__ == "__main__":
    print("--- Generating Training Images ---")
    generate_captcha_images(num_train_images, output_train_dir)
    print("\n--- Generating Testing Images ---")
    generate_captcha_images(num_test_images, output_test_dir)
    print("\nImage generation complete.")