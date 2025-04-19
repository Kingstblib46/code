import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from PIL import Image

import capcha.common as common
from capcha.model import MyModel
from capcha.dataset import CaptchaDataset
import capcha.one_hot as one_hot

# --- Prediction Parameters ---
test_dataset_path = './datasets/test'
model_load_path = './model.pth'
batch_size = 128
num_workers = 4
num_samples_to_show = 10

def predict():
    # 1. Load test data
    print(f"Loading testing data from {test_dataset_path}...")
    try:
        test_dataset = CaptchaDataset(root_dir=test_dataset_path)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if common.device == 'cuda' else False)
        print(f"Test data loaded. Dataset size: {len(test_dataset)}, Batches: {len(test_dataloader)}")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        print("Please ensure you have run generate_images.py and the test dataset exists.")
        return

    # 2. Load trained model
    print(f"Loading trained model from {model_load_path}...")
    try:
        model = MyModel().to(common.device)
        model.load_state_dict(torch.load(model_load_path, map_location=common.device))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_load_path}. Please train the model first by running train.py.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Make predictions and evaluate
    correct = 0
    total = 0
    all_results = []

    print("Starting prediction...")
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Predicting")
        for imgs, targets in progress_bar:
            imgs = imgs.to(common.device)
            targets = targets.to(common.device)

            outputs = model(imgs)

            # Reshape for decoding
            outputs_reshaped = outputs.view(-1, common.captcha_size, common.num_classes)
            targets_reshaped = targets.view(-1, common.captcha_size, common.num_classes)

            predict_labels = one_hot.vec_to_text(outputs_reshaped)
            true_labels = one_hot.vec_to_text(targets_reshaped)

            current_batch_size = len(true_labels)
            for i in range(current_batch_size):
                is_correct = (predict_labels[i] == true_labels[i])
                if is_correct:
                    correct += 1
                all_results.append({
                    "true": true_labels[i],
                    "pred": predict_labels[i],
                    "correct": is_correct
                })

            total += current_batch_size

            # Update progress bar
            accuracy = (correct / total) * 100 if total > 0 else 0.0
            progress_bar.set_postfix({'acc': f'{accuracy:.2f}%'})

    # 4. Output overall accuracy
    final_accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"\nPrediction finished.")
    print(f"Total tested: {total}")
    print(f"Correctly predicted: {correct}")
    print(f"Accuracy: {final_accuracy:.2f}%")

    # 5. Visualize sample predictions
    print(f"\nShowing {num_samples_to_show} sample predictions...")
    random_indices = random.sample(range(len(test_dataset)), k=min(num_samples_to_show, len(test_dataset)))

    fig, axes = plt.subplots(nrows=(num_samples_to_show + 1) // 2, ncols=2, figsize=(10, num_samples_to_show * 1.5))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        img_path = test_dataset.image_paths[idx]
        img_tensor, label_vec_flat = test_dataset[idx]

        img_tensor_batch = img_tensor.unsqueeze(0).to(common.device)
        output = model(img_tensor_batch)
        output_reshaped = output.view(1, common.captcha_size, common.num_classes)
        predicted_label = one_hot.vec_to_text(output_reshaped)

        label_reshaped = label_vec_flat.view(common.captcha_size, common.num_classes)
        true_label = one_hot.vec_to_text(label_reshaped)

        img_display = Image.open(img_path)
        ax = axes[i]
        ax.imshow(img_display)
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}", color='green' if true_label == predicted_label else 'red')
        ax.axis('off')

    if len(random_indices) % 2 != 0 and len(axes) > len(random_indices):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predict()