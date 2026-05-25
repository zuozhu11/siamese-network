# siamese-network
A siamese network that is fine-tuned by a  MobileNet_V3_Small net

# Siamese Network for Image Similarity using PyTorch

This project provides a complete and concise implementation of a Siamese Network for image similarity, built with PyTorch. It is designed to be an excellent starting point for beginners interested in metric learning, image similarity, and fine-tuning pre-trained models.

The script trains a model to determine if two images from the FashionMNIST dataset are from the same class.

## Key Features

*   **Complete & Runnable**: The entire workflow—from data loading and model definition to training and validation—is contained within a single, easy-to-run Python script (`model.py`).
*   **Simple & Beginner-Friendly**: The code is written with clarity in mind, making it perfect for educational purposes. It serves as a practical, hands-on introduction to Siamese networks.
*   **Fine-Tuning Approach**: This project is a clear example of **fine-tuning**. It leverages a **pre-trained MobileNetV3** model as its backbone for feature extraction and only trains a small, custom-built fully connected head. This is a common and effective technique in modern deep learning.
*   **Best Model Saving**: The script automatically monitors the validation loss and saves the best-performing model weights to `best_model.pth`, a crucial practice in model training.

## How It Works

1.  **Embedding Network**: A pre-trained `MobileNetV3` (small version) is used as the base feature extractor. Its final classification layer is replaced with a custom MLP to produce a 128-dimensional embedding.
2.  **Siamese Network**: The `SiameseNet` class takes two images, passes each through the same embedding network, and outputs their respective embeddings.
3.  **Dataset**: A custom `SiameseMNIST` dataset class wraps the standard FashionMNIST dataset. For each item, it generates a pair of images, which can be either from the same class (positive pair) or different classes (negative pair).
4.  **Loss Function**: The model is trained using **Contrastive Loss**. This loss function pushes embeddings from similar images closer together and pulls embeddings from dissimilar images further apart.
5.  **Training Loop**: A standard PyTorch training loop is implemented in the `fit` function, which includes training, validation, learning rate scheduling, and saving the best model based on validation loss.

## How to Run

### Prerequisites

Make sure you have the following libraries installed:

```bash
pip install torch torchvision numpy Pillow
```

### Execution

Simply run the script from your terminal:

```bash
python model.py
```

The script will automatically download the FashionMNIST dataset, start the training process, and save the `best_model.pth` file in the same directory.
