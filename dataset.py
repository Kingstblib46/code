import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import common
import one_hot

class CaptchaDataset(Dataset):
    def __init__(self, root_dir):
        super(CaptchaDataset, self).__init__()
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        if not self.image_paths:
             raise FileNotFoundError(f"No PNG images found in directory: {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((common.image_height, common.image_width)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # From filename get label (format: label_timestamp_index.png)
        try:
            image_name = os.path.basename(image_path)
            label_text = image_name.split('_')[0]
            if len(label_text) != common.captcha_size:
                return self.__getitem__((index + 1) % len(self))
        except IndexError:
             return self.__getitem__((index + 1) % len(self))

        try:
            img = Image.open(image_path)
            img_tensor = self.transform(img)
        except Exception:
            return self.__getitem__((index + 1) % len(self))

        # Convert label text to one-hot encoded vector
        try:
            label_vec = one_hot.text_to_vec(label_text)
            # Flatten for MultiLabelSoftMarginLoss input
            label_vec_flat = label_vec.view(1, -1).squeeze() # Shape: (captcha_size * num_classes)
        except ValueError:
            return self.__getitem__((index + 1) % len(self))

        return img_tensor, label_vec_flat

# Test data loading
if __name__ == '__main__':
    train_dataset_path = './datasets/train'
    if not os.path.exists(train_dataset_path) or not os.listdir(train_dataset_path):
         print(f"Error: Training dataset directory '{train_dataset_path}' is empty or does not exist.")
         print("Please run generate_images.py first to create dataset.")
    else:
        try:
            train_dataset = CaptchaDataset(root_dir=train_dataset_path)
            print(f"Dataset size: {len(train_dataset)}")

            batch_size = 64
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            img_batch, label_batch = next(iter(train_dataloader))

            print(f"Image batch shape: {img_batch.shape}") # (batch_size, 1, height, width)
            print(f"Label batch shape: {label_batch.shape}") # (batch_size, captcha_size * num_classes)

            # Visualize a sample
            import matplotlib.pyplot as plt
            idx_to_show = 0
            plt.imshow(img_batch[idx_to_show].squeeze(), cmap='gray')
            # Reshape back to (captcha_size, num_classes) for decoding
            label_reshaped = label_batch[idx_to_show].view(common.captcha_size, common.num_classes)
            decoded_label = one_hot.vec_to_text(label_reshaped)
            plt.title(f"Label: {decoded_label}")
            plt.show()

        except Exception as e:
             print(f"An error occurred during dataset loading test: {e}")