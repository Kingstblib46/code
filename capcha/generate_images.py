import os
import random
import time
from captcha.image import ImageCaptcha
import capcha.common as common

num_train_images = 50000
num_test_images = 10000
output_train_dir = "./datasets/train"
output_test_dir = "./datasets/test"

def generate_captcha_images(num_images, output_dir):
    """Generate captcha images and save to directory"""
    image_gen = ImageCaptcha(width=common.image_width, height=common.image_height)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_images} images to {output_dir}...")

    from tqdm import tqdm
    for i in tqdm(range(num_images), desc=f"Generating to {os.path.basename(output_dir)}"):
        image_val = "".join(random.sample(common.captcha_array, common.captcha_size))
        image_name = f"{image_val}_{int(time.time() * 1000)}_{i}.png"
        image_path = os.path.join(output_dir, image_name)

        try:
            image_gen.write(image_val, image_path)
        except Exception as e:
            print(f"Error writing file {image_path}: {e}")

if __name__ == "__main__":
    print("--- Generating Training Images ---")
    generate_captcha_images(num_train_images, output_train_dir)
    print("\n--- Generating Testing Images ---")
    generate_captcha_images(num_test_images, output_test_dir)
    print("\nImage generation complete.")