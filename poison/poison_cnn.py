import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import os
import time
from datetime import datetime

# Define result directories (using relative paths)
# These will be created relative to the script's execution directory (e.g., ~/poison/res/log)
LOG_DIR = "./res/log"
IMG_DIR = "./res/img"

# Ensure base directories exist relative to the current working directory
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Generate timestamp string for filenames
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define AlexNet structure (adapted for MNIST 28x28 grayscale)
# Based on the provided slides/PDF (sources [70]-[78])
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # MNIST images are 1x28x28 (Channel x Height x Width)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Output: 32x28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: 32x14x14
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Output: 64x14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: 64x7x7
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # Output: 128x7x7
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)# Output: 256x7x7
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)# Output: 256x7x7
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: 256x3x3 (Assuming 7x7 input and kernel/stride=2)
        self.relu3 = nn.ReLU()

        # Calculate the flattened size after conv/pool layers
        # Output size = 256 * 3 * 3 = 2304
        self.fc6 = nn.Linear(256 * 3 * 3, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        # Flatten the feature map
        x = x.view(-1, 256 * 3 * 3) # Corrected flatten dimension
        x = self.fc6(x)
        x = F.relu(x)
        # Consider adding Dropout here if needed, e.g., x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc7(x)
        x = F.relu(x)
        # Consider adding Dropout here
        x = self.fc8(x)
        return x

# Function to select a subset of the dataset
# Based on source [7, 79]
def select_subset(dataset, ratio=0.1):
    """Selects a random subset of the dataset."""
    subset_size = int(len(dataset) * ratio)
    if subset_size > len(dataset): # Ensure subset size doesn't exceed dataset length
      subset_size = len(dataset)
    elif subset_size == 0 and ratio > 0: # Ensure at least one sample if ratio > 0
        subset_size = 1

    if subset_size == 0:
        print("Warning: Subset ratio resulted in zero samples.")
        return Subset(dataset, [])

    indices = np.random.choice(range(len(dataset)), subset_size, replace=False)
    return Subset(dataset, indices)

# Custom Dataset for poisoned data
class PoisonedDataset(Dataset):
    """A dataset wrapper to hold poisoned and clean data."""
    def __init__(self, data_list):
        self.data = data_list # List of (image_tensor, label) tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Function to create poisoned and clean datasets (Label Flipping)
# Based on source [10, 85]
def fetch_poisoned_datasets(full_dataset, train_subset_indices, poison_ratio, num_classes=10):
    """
    Creates poisoned and clean datasets based on the specified ratio.
    Implements label flipping: randomly assigns a wrong label to poisoned samples.
    """
    poison_trainset_list = []
    clean_trainset_list = []

    # Group images by class from the selected subset
    character = [[] for _ in range(num_classes)]
    # Store original label for potential targeted flipping
    # No need to store original_labels explicitly if just doing random flip
    for index in train_subset_indices:
        # Ensure index is within bounds
        if index < 0 or index >= len(full_dataset):
             print(f"Warning: Index {index} out of bounds for full_dataset (size {len(full_dataset)}). Skipping.")
             continue
        img, label = full_dataset[index] # Get image and original label
        # Ensure label is within bounds
        if label < 0 or label >= num_classes:
            print(f"Warning: Label {label} out of bounds for num_classes ({num_classes}). Skipping sample.")
            continue
        character[label].append(img)


    # Process each class
    for i, data in enumerate(character): # i is the original label (0-9)
        if not data: # Skip if no samples for this class in the subset
            continue

        num_samples = len(data)
        num_poison_train_inputs = int(num_samples * poison_ratio)
        if num_poison_train_inputs > num_samples: # Cap at available samples
            num_poison_train_inputs = num_samples

        # Indices for this class's samples
        class_indices = list(range(num_samples))
        random.shuffle(class_indices)

        poison_indices = class_indices[:num_poison_train_inputs]
        clean_indices = class_indices[num_poison_train_inputs:]

        # Create poisoned samples
        for idx in poison_indices:
            img = data[idx]
            # Convert PIL Image to numpy array, normalize, and convert to tensor
            img_np = np.array(img) / 255.0
            img_tensor = torch.from_numpy(img_np).float() # Ensure float tensor

            # --- Label Flipping Logic ---
            # Strategy 1: Random incorrect label (as in example source [85])
            possible_targets = list(range(num_classes))
            if num_classes > 1: # Ensure there are other labels to choose from
                possible_targets.remove(i) # Remove the original label
            if not possible_targets: # Handle edge case of single class
                target_label = i # Cannot flip if only one class
            else:
                target_label = random.choice(possible_targets)

            # Strategy 2: Next label (0->1, 1->2, ..., 9->0) (as requested in source [21])
            # target_label = (i + 1) % num_classes

            poison_trainset_list.append((img_tensor, target_label))

        # Create clean samples
        for idx in clean_indices:
            img = data[idx]
            img_np = np.array(img) / 255.0
            img_tensor = torch.from_numpy(img_np).float()
            clean_trainset_list.append((img_tensor, i)) # Use the original label

    print(f"Created {len(poison_trainset_list)} poisoned samples and {len(clean_trainset_list)} clean samples for training.")
    if not poison_trainset_list and not clean_trainset_list and len(train_subset_indices) > 0:
         print("Warning: No training samples were generated. Check subset selection and poisoning logic.")

    return PoisonedDataset(poison_trainset_list), PoisonedDataset(clean_trainset_list)


# --- Plotting Functions ---
# Based on source [8, 9, 80, 83]
def plot_image_predictions(model, dataset, device, correct=True, num_images=10, title_prefix="", base_img_dir=IMG_DIR, timestamp=None):
    """Plots correctly or incorrectly classified images and saves to base_img_dir."""
    # Use passed timestamp or global timestamp
    ts_prefix = timestamp or TIMESTAMP
    
    model.eval()
    classified_imgs = []
    count = 0

    if len(dataset) == 0:
        print(f"Cannot plot predictions: Dataset is empty.")
        return

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        if count >= num_images:
            break

        img, label = dataset[idx]
        # Ensure image has correct dimensions [C, H, W] and add batch dim [B, C, H, W]
        if isinstance(img, np.ndarray): # If loaded from custom dataset directly
             img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device) # Add Channel and Batch dim
        elif isinstance(img, torch.Tensor):
             # Ensure tensor has channel dim before unsqueezing batch dim
             if img.dim() == 2: # H, W -> add C dim
                 img = img.unsqueeze(0)
             if img.dim() == 3: # C, H, W -> add B dim
                 img_tensor = img.float().unsqueeze(0).to(device)
             else:
                 print(f"Warning: Unexpected tensor dimension {img.dim()}. Skipping prediction plot.")
                 continue
        else: # Assuming PIL image from standard dataset
             img_tensor = transforms.ToTensor()(img).float().unsqueeze(0).to(device)


        with torch.no_grad():
            pred = model(img_tensor)
            pred_label = torch.argmax(pred, dim=1).item()

        is_correct = (pred_label == label)
        if (correct and is_correct) or (not correct and not is_correct):
            # Ensure image is suitable for plotting (squeeze batch/channel dims, move to CPU, convert to numpy)
            img_display = img_tensor.cpu().squeeze().numpy()
            classified_imgs.append((img_display, label, pred_label))
            count += 1

    if not classified_imgs:
        print(f"No {'correct' if correct else 'incorrect'} images found to plot for {title_prefix[:-1]}.")
        return

    plt.figure(figsize=(10, max(5, count * 1))) # Adjust figure size
    plt.suptitle(f"{title_prefix}{'Correct' if correct else 'Misclassified'} Examples", fontsize=16)
    # Calculate rows needed, ensuring at least 1 row
    rows = max(1, (count + 1) // 2 )
    for i, (img_display, true_label, pred_label) in enumerate(classified_imgs):
        plt.subplot(rows, 2, i + 1)
        plt.imshow(img_display, cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # --- Save Figure to specified directory ---
    fig_filename_base = f"{ts_prefix}_{title_prefix}{'correct' if correct else 'misclassified'}_examples.png"
    # Construct full path using os.path.join
    fig_save_path = os.path.join(base_img_dir, fig_filename_base)
    # Ensure directory exists
    os.makedirs(base_img_dir, exist_ok=True) # Redundant if created at start, but safe

    try:
        plt.savefig(fig_save_path)
        print(f"Saved prediction examples to {fig_save_path}")
    except Exception as e:
        print(f"Error saving figure {fig_save_path}: {e}")
    # plt.show() # Optionally display plot immediately
    plt.close() # Close figure to free memory


def plot_metrics(epochs, train_losses, test_accuracies, poison_ratio, args_dict, base_img_dir=IMG_DIR, timestamp=None):
    """Plots training loss and test accuracy curves and saves to base_img_dir."""
    if not epochs or not train_losses or not test_accuracies:
         print("Warning: Cannot plot metrics due to empty data lists.")
         return

    # Use passed timestamp or global timestamp
    ts_prefix = timestamp or TIMESTAMP
    
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.title(f'Training Loss (Poison: {poison_ratio:.2f})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, test_accuracies, 'ro-', label='Clean Test Accuracy')
    plt.title(f'Clean Test Acc (Poison: {poison_ratio:.2f})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100) # Accuracy is between 0 and 100
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # --- Save Figure to specified directory ---
    # Create filename incorporating key parameters with timestamp
    fig_filename_base = f"{ts_prefix}_metrics_poison_{args_dict['poison_ratio']:.2f}_sub_{args_dict['subset_ratio']:.2f}_lr_{args_dict['lr']}_epochs_{args_dict['epochs']}.png"
    fig_save_path = os.path.join(base_img_dir, fig_filename_base)
    # Ensure directory exists
    os.makedirs(base_img_dir, exist_ok=True)

    try:
        plt.savefig(fig_save_path)
        print(f"Saved metrics plot to {fig_save_path}")
    except Exception as e:
        print(f"Error saving figure {fig_save_path}: {e}")

    # plt.show() # Optionally display plot immediately
    plt.close() # Close figure to free memory

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Data Poisoning Research - Label Flipping Attack")
    parser.add_argument('--poison_ratio', type=float, default=0.1, help='Fraction of training data to poison (default: 0.1)')
    parser.add_argument('--subset_ratio', type=float, default=0.1, help='Fraction of original dataset to use (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA training')
    parser.add_argument('--plot_examples', action='store_true', default=False, help='Plot example classifications')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers (default: 2)')

    args = parser.parse_args()
    args_dict = vars(args) # Convert args to dictionary for easier use in filenames

    # Validate ratios
    if not 0.0 <= args.poison_ratio <= 1.0:
        raise ValueError("Poison ratio must be between 0.0 and 1.0")
    if not 0.0 < args.subset_ratio <= 1.0:
         # Allow 0 for testing, but usually want > 0
         if args.subset_ratio == 0.0:
             print("Warning: subset_ratio is 0, no data will be used.")
         else:
            raise ValueError("Subset ratio must be between 0.0 (exclusive, usually) and 1.0")


    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Potentially disable benchmark for full determinism, might cost performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    print(f"Arguments: {args}")


    # --- 1. Load Data ---
    print("Loading MNIST dataset...")
    # Load the full datasets (without transforms initially, as poisoning function handles numpy conversion)
    try:
        # Specify data root relative to script location or an absolute path
        data_root = './data' # Store original data in ./data relative to script
        trainset_all = datasets.MNIST(root=data_root, train=True, download=True)
        testset_all = datasets.MNIST(root=data_root, train=False, download=True)
    except Exception as e:
        print(f"Error downloading or loading MNIST dataset: {e}")
        print("Please check your internet connection or dataset path.")
        exit()

    # Select subsets based on ratio
    print(f"Selecting {args.subset_ratio*100:.1f}% subset of data...")
    trainset_subset = select_subset(trainset_all, ratio=args.subset_ratio)
    testset_subset = select_subset(testset_all, ratio=args.subset_ratio)
    train_subset_indices = trainset_subset.indices
    test_subset_indices = testset_subset.indices

    if not hasattr(train_subset_indices, 'size') or not train_subset_indices.size > 0:
         print("Error: No training samples selected based on subset ratio. Exiting.")
         exit()
    if not hasattr(test_subset_indices, 'size') or not test_subset_indices.size > 0:
         print("Error: No test samples selected based on subset ratio. Exiting.")
         exit()


    # --- 2. Prepare Poisoned Training Data ---
    print(f"Preparing poisoned dataset with poison ratio: {args.poison_ratio}...")
    poison_trainset, clean_trainset = fetch_poisoned_datasets(
        full_dataset=trainset_all,
        train_subset_indices=train_subset_indices,
        poison_ratio=args.poison_ratio,
        num_classes=10
    )

    # Combine poisoned and clean samples for the final training set
    all_train_data = poison_trainset.data + clean_trainset.data
    if not all_train_data:
        print("Error: Combined training dataset is empty. Exiting.")
        exit()

    random.shuffle(all_train_data) # Shuffle combined data
    combined_trainset = PoisonedDataset(all_train_data)

    # Create DataLoader for the combined training set
    train_loader = DataLoader(
        combined_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if use_cuda else 0, # Use workers only with CUDA potentially
        pin_memory=use_cuda # pin_memory often helps with GPU transfer speed
    )
    print(f"Total training samples: {len(combined_trainset)}")


    # --- 3. Prepare Clean Test Data ---
    print("Preparing clean test dataset...")
    # Create a clean test set from the selected test subset indices
    clean_testset_list = []
    for index in test_subset_indices:
         img, label = testset_all[index]
         img_np = np.array(img) / 255.0
         img_tensor = torch.from_numpy(img_np).float()
         clean_testset_list.append((img_tensor, label))

    if not clean_testset_list:
        print("Error: Clean test dataset is empty. Exiting.")
        exit()

    clean_testset = PoisonedDataset(clean_testset_list)
    test_loader = DataLoader(
        clean_testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if use_cuda else 0,
        pin_memory=use_cuda
    )
    print(f"Clean test samples: {len(clean_testset)}")


    # --- 4. Initialize Model, Loss, Optimizer ---
    print("Initializing model...")
    model = AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 5. Training Loop ---
    print("Starting training...")
    start_time = time.time()
    train_losses = []
    test_accuracies = []
    # --- Construct Log file path using RELATIVE path base with timestamp ---
    log_filename = f"{TIMESTAMP}_training_log_poison_{args.poison_ratio:.2f}_sub_{args.subset_ratio:.2f}_lr_{args.lr}_epochs_{args.epochs}.txt"
    log_file_path = os.path.join(LOG_DIR, log_filename)
    # Ensure directory exists (relative path)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Use try-finally to ensure plots are generated even if training fails
    try:
        with open(log_file_path, "w") as log_file:
            log_file.write(f"Arguments: {args}\n")
            log_file.write(f"Device: {device}\n\n")
            log_file.flush() # Ensure header is written

            for epoch in range(args.epochs):
                model.train()
                running_loss = 0.0
                batch_count = 0
                epoch_start_time = time.time() # Time each epoch

                for i, (inputs, labels) in enumerate(train_loader):
                    # Manually add channel dimension for grayscale MNIST
                    # Ensure tensor format [B, C, H, W]
                    if inputs.dim() == 3: # B, H, W -> B, 1, H, W
                        inputs = inputs.unsqueeze(1)
                    elif inputs.dim() != 4: # Check if not already 4D
                         print(f"Error: Unexpected input dimension {inputs.dim()} in training batch {i}. Skipping batch.")
                         continue

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    batch_count += 1
                    # Print progress less frequently for larger datasets/epochs
                    print_freq = max(1, len(train_loader) // 10) # Print ~10 times per epoch
                    if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader): # Also print on last batch
                        print(f'\rEpoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}', end='')

                # Ensure the line is cleared after the epoch loop finishes printing progress
                print(' ' * 40, end='\r') # Clear the line with spaces
                print() # Moves to the next line after epoch completion

                if batch_count == 0:
                    print(f"Warning: Epoch {epoch+1} had no batches processed.")
                    epoch_loss = 0.0 # Avoid division by zero
                else:
                    epoch_loss = running_loss / batch_count
                train_losses.append(epoch_loss)
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time

                print(f"Epoch {epoch+1}/{args.epochs} completed. Avg Loss: {epoch_loss:.4f}. Duration: {epoch_duration:.2f}s")
                log_file.write(f"Epoch: {epoch+1}, Avg Loss: {epoch_loss:.4f}, Duration: {epoch_duration:.2f}s\n")
                log_file.flush()

                # --- 6. Evaluation (on Clean Test Set) ---
                model.eval()
                correct = 0
                total = 0
                eval_start_time = time.time()
                with torch.no_grad():
                    for images, labels in test_loader:
                        # Manually add channel dimension
                        if images.dim() == 3: # B, H, W -> B, 1, H, W
                            images = images.unsqueeze(1)
                        elif images.dim() != 4:
                            print(f"Error: Unexpected image dimension {images.dim()} in evaluation. Skipping batch.")
                            continue

                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                eval_end_time = time.time()
                eval_duration = eval_end_time - eval_start_time

                if total == 0:
                     print(f"Warning: Epoch {epoch+1} evaluation had no samples.")
                     clean_accuracy = 0.0 # Avoid division by zero
                else:
                    clean_accuracy = 100 * correct / total
                test_accuracies.append(clean_accuracy)
                print(f"Epoch {epoch+1}: Clean Test Accuracy = {clean_accuracy:.2f}%. Eval Duration: {eval_duration:.2f}s")
                log_file.write(f"Epoch: {epoch+1}, Clean Test Accuracy: {clean_accuracy:.2f}%, Eval Duration: {eval_duration:.2f}s\n")
                log_file.flush()

            # --- End of Training Loop (Inside With Block) ---
            end_time = time.time()
            training_duration = end_time - start_time
            log_file.write(f"\nTraining finished. Total time: {training_duration:.2f} seconds\n")
            if test_accuracies: # Ensure list is not empty
                log_file.write(f"Final Clean Test Accuracy: {test_accuracies[-1]:.2f}%\n")
            log_file.flush()

        # 'with' block ends here, log_file is now closed.
        print(f"\nTraining finished. Total time: {training_duration:.2f} seconds")
        if test_accuracies:
             print(f"Final Clean Test Accuracy: {test_accuracies[-1]:.2f}%")
        print(f"Log saved to: {log_file_path}")

    finally: # Use finally to ensure plots are generated even if errors occur during training
        # --- 7. Visualization ---
        print("Generating plots...")
        plot_metrics(args.epochs, train_losses, test_accuracies, args.poison_ratio, args_dict, base_img_dir=IMG_DIR)

        if args.plot_examples:
            print("Plotting classification examples...")
            # Create a unique prefix for example plots based on parameters
            example_plot_prefix = f"Psn{args.poison_ratio:.2f}_Sub{args.subset_ratio:.2f}_Lr{args.lr}_Ep{args.epochs}_"
            # Plot examples from the clean test set
            plot_image_predictions(model, clean_testset, device, correct=True, num_images=10, title_prefix=example_plot_prefix, base_img_dir=IMG_DIR)
            plot_image_predictions(model, clean_testset, device, correct=False, num_images=10, title_prefix=example_plot_prefix, base_img_dir=IMG_DIR)

        print("\nExperiment complete.")
        print("--------------------------------------------------")
        print("Further Research Directions (Based on provided documents):")
        print("- Modify `fetch_poisoned_datasets` for different label flipping strategies (e.g., 0->1, 9->0).")
        print("- Implement Backdoor Attacks (e.g., adding triggers): See BackdoorBox library.")
        print("- Implement Clean-Label Attacks (e.g., Feature Collision, MetaPoison): See ART library and MetaPoison paper.")
        print("- Evaluate Attack Success Rate (ASR) for targeted attacks/backdoors (requires poisoned test set).")
        print("- Implement and evaluate defenses mentioned in the research paper.")
        print("--------------------------------------------------")