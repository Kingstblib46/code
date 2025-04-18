# CNN-Based CAPTCHA Recognition using PyTorch

This repository contains a Python project for recognizing simple text-based CAPTCHA images using a Convolutional Neural Network (CNN) built with PyTorch. The code allows for generating CAPTCHA datasets, training a CNN model, and evaluating its performance.

This implementation is based on the practical exercises found in course materials for "Chapter 3: Security Applications of Convolutional Neural Networks"[cite: 29, 64].

## Features

* **CAPTCHA Data Generation:** Script to generate custom datasets of CAPTCHA images with configurable size and character sets[cite: 69].
* **CNN Model:** A defined CNN architecture suitable for CAPTCHA recognition[cite: 39, 64].
* **Training:** Script to train the CNN model on the generated dataset, utilizing GPU acceleration (CUDA) if available[cite: 31].
* **Prediction & Evaluation:** Script to evaluate the trained model on a test set and visualize prediction results[cite: 32, 71].
* **Logging:** Includes a shell script to run predictions and save the terminal output to a timestamped log file.

## Project Structure

```
capcha_recognition/
├── datasets/
│   ├── train/    # Populated by generate_images.py
│   └── test/     # Populated by generate_images.py
├── log/          # Created by run_predict.sh for logs
├── common.py     # Shared variables and configurations
├── dataset.py    # PyTorch Dataset class for CAPTCHA images
├── generate_images.py # Script to generate CAPTCHA dataset
├── model.py      # CNN model definition (MyModel)
├── one_hot.py    # Helper functions for one-hot encoding/decoding labels
├── predict.py    # Script to run predictions with a trained model
├── train.py      # Script to train the CNN model
├── requirements.txt # Python package dependencies
├── run_predict.sh   # Shell script to run predict.py and log output
└── model.pth     # Saved model weights (after training)
```

## Requirements

The required Python packages are listed in `requirements.txt`. Key dependencies include:

* PyTorch (`torch`, `torchvision`) [cite: 38]
* `captcha` [cite: 38]
* `tqdm` [cite: 38]
* `numpy`
* `matplotlib`
* `Pillow`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd capcha_recognition
    ```

2.  **Create a Conda Environment (Recommended):**
    It's highly recommended to use a dedicated environment (e.g., using Conda or venv). The code was tested successfully after downgrading NumPy due to compatibility issues with PyTorch compiled against NumPy 1.x.
    ```bash
    # Create environment (using Python 3.12 as per logs, adjust if needed)
    conda create --name capcha python=3.12 -y

    # Activate the environment
    conda activate capcha
    ```

3.  **Install Dependencies:**
    Install the required packages using pip and the provided `requirements.txt`. **Crucially, ensure NumPy version < 2 is installed to avoid compatibility issues observed during testing.**
    ```bash
    # Ensure numpy<2 is specified or handle it manually:
    pip uninstall numpy -y
    pip install "numpy<2"
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` already pins `numpy<2`, the separate install/uninstall might not be needed)*

4.  **Create Directories:**
    The `datasets/train` and `datasets/test` directories are needed for data generation. The `log` directory is needed for the `run_predict.sh` script.
    ```bash
    mkdir -p datasets/train datasets/test log
    ```

## Usage

**Important:** Ensure your Conda environment (`capcha` or your chosen name) is activated before running any scripts (`conda activate capcha`).

1.  **Generate CAPTCHA Data:**
    Run the script to create training and testing images. You can adjust the number of images generated within the script itself.
    ```bash
    python generate_images.py
    ```
    This will populate the `datasets/train` and `datasets/test` folders.

2.  **Train the Model:**
    Start the training process. The script will use the GPU if available (CUDA). Adjust epochs, batch size, etc., within `train.py` for potentially better results, especially with powerful hardware like an RTX 4090.
    ```bash
    # Ensure environment is active!
    # Use the specific python interpreter if 'python' doesn't resolve correctly
    # /path/to/your/conda/envs/capcha/bin/python train.py
    python train.py
    ```
    This will save the trained model weights to `model.pth`.

3.  **Evaluate the Model / Make Predictions:**
    Run predictions on the test set using the saved model.
    ```bash
    # Ensure environment is active!
    python predict.py
    ```
    This will print the overall accuracy and show a window with some sample predictions[cite: 33].

4.  **Run Prediction with Logging:**
    Use the shell script to run predictions and save the output to the `log` directory.
    ```bash
    # Make the script executable (only needed once)
    chmod +x run_predict.sh

    # Run the script (ensure conda env is active)
    ./run_predict.sh
    ```
    Check the `log/` directory for timestamped log files containing the output.

## Configuration

Key parameters can be adjusted directly within the Python scripts:

* `generate_images.py`: `num_train_images`, `num_test_images`
* `train.py`: `epochs`, `batch_size`, `learning_rate`, `num_workers`
* `common.py`: `captcha_array` (character set), `captcha_size` (length), image dimensions (`image_height`, `image_width`)
* `run_predict.sh`: `PYTHON_EXECUTABLE` (path to correct python interpreter)

## Model Architecture

The CNN model (`model.py`) consists of[cite: 64, 65, 66, 67]:
* Four convolutional blocks, each containing:
    * `Conv2d` layer (with padding)
    * `BatchNorm2d`
    * `ReLU` activation
    * `MaxPool2d`
* A final classifier section with:
    * `Flatten` layer
    * Two `Linear` layers (with `Dropout` and `ReLU` in between)
* The output layer produces scores for each possible character at each position in the CAPTCHA[cite: 65].

## Results

Based on the provided materials, the model achieved an accuracy of around 84.80% on the test set after training for 10 epochs[cite: 34]. Your results may vary depending on the amount of training data generated, hyperparameters (epochs, batch size, learning rate), and random initialization.

*(You can replace this section with your own results after running the experiments)*

## License

*(Consider adding a license file, e.g., MIT License)*

```text
MIT License

Copyright (c) [Year] [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgements

* This project is based on course materials provided by Prof. Li Jian, School of Cyberspace Security, Beijing University of Posts and Telecommunications (BUPT)[cite: 29].

```

Remember to:
* Replace `<your-repo-url>` with the actual URL when you create the repository.
* Replace `[Year]` and `[Your Name/Organization]` in the License section if you choose to include it.
* Update the "Results" section with your findings if desired.