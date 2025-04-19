import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import capcha.common as common
from capcha.model import MyModel
from capcha.dataset import CaptchaDataset
import capcha.one_hot as one_hot

# --- Training Parameters ---
train_dataset_path = './datasets/train'
model_save_path = './model.pth'
learning_rate = 0.001
epochs = 10
batch_size = 64
num_workers = 4

def train():
    # 1. Load data
    print(f"Loading training data from {train_dataset_path}...")
    try:
        train_dataset = CaptchaDataset(root_dir=train_dataset_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if common.device == 'cuda' else False)
        print(f"Data loaded. Dataset size: {len(train_dataset)}, Batches: {len(train_dataloader)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have run generate_images.py and the dataset exists.")
        return

    # 2. Initialize model, loss function, optimizer
    print("Initializing model...")
    model = MyModel().to(common.device)

    # MultiLabelSoftMarginLoss expects input: (N, C) where C is captcha_size * num_classes
    loss_fn = nn.MultiLabelSoftMarginLoss().to(common.device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    # 3. Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")

        for step, (imgs, targets) in progress_bar:
            imgs = imgs.to(common.device)       # (B, 1, H, W)
            targets = targets.to(common.device) # (B, captcha_size * num_classes)

            optimizer.zero_grad()

            outputs = model(imgs) # (B, captcha_size * num_classes)

            loss = loss_fn(outputs, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            outputs_reshaped = outputs.view(-1, common.captcha_size, common.num_classes)
            targets_reshaped = targets.view(-1, common.captcha_size, common.num_classes)

            predict_labels = one_hot.vec_to_text(outputs_reshaped)
            true_labels = one_hot.vec_to_text(targets_reshaped)

            current_batch_size = len(true_labels)
            for i in range(current_batch_size):
                 if predict_labels[i] == true_labels[i]:
                     correct_predictions += 1
            total_predictions += current_batch_size
            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})

        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # 4. Save model
    print(f"Training finished. Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    train()