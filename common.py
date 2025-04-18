import torch

# Captcha character set
captcha_array = list("0123456789abcdefghijklmnopqrstuvwxyz")
# Captcha length
captcha_size = 4
# Character set size
num_classes = len(captcha_array)

# Device selection (use CUDA GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image dimensions
image_height = 60
image_width = 160

print(f"Using device: {device}")
print(f"Captcha characters: {''.join(captcha_array)}")
print(f"Captcha size: {captcha_size}")
print(f"Number of classes: {num_classes}")