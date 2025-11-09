# Simple CNN with PyTorch — MNIST Example

This notebook demonstrates how to build, train, and evaluate a basic Convolutional Neural Network (CNN) using PyTorch on the well-known MNIST dataset of handwritten digits.

## Table of Contents

1.  **Setup**: Installation of necessary libraries (PyTorch, torchvision, numpy, matplotlib).
2.  **Imports and Device Configuration**: Importing core libraries and setting up the computational device (CPU/GPU).
3.  **Data Loading and Preprocessing**: Loading the MNIST dataset using `torchvision.datasets`, applying transformations (`ToTensor`, `Normalize`), and preparing `DataLoader`s for efficient batching.
4.  **Model Definition**: Defining a simple CNN architecture using `torch.nn.Module`, consisting of convolutional layers, ReLU activations, Max Pooling, flattening, and fully-connected layers for classification.
5.  **Loss Function and Optimizer**: Setting up `nn.CrossEntropyLoss` and the `Adam` optimizer.
6.  **Training and Evaluation Utilities**: Implementing `train_one_epoch` and `evaluate` functions to manage the training and validation loops.
7.  **Training Execution**: Running the training process for a specified number of epochs and logging performance metrics.
8.  **Plotting Training Curves**: Visualizing the training and validation loss and accuracy over epochs using `matplotlib`.
9.  **Visualizing Predictions**: Displaying sample images from the test set along with their true labels and the model's predictions to qualitatively assess performance.

## What You'll Learn

*   How to load standard datasets like MNIST using `torchvision.datasets`.
*   Effective use of `torch.utils.data.DataLoader` for batching, shuffling, and parallel data loading.
*   Defining custom neural network architectures with `torch.nn.Module` and its layers (`Conv2d`, `ReLU`, `MaxPool2d`, `Linear`, `Flatten`).
*   Implementing a full training and evaluation loop in PyTorch.
*   Monitoring model performance through loss and accuracy plots.
*   Visualizing model predictions to understand its behavior.

## How to Run

1.  **Open in Google Colab**: Click the "Open in Colab" badge (if available) or upload the `.ipynb` file to your Google Drive and open it with Colaboratory.
2.  **Run All Cells**: Execute all cells in sequential order. The notebook is designed to be run from top to bottom.
3.  **GPU Acceleration (Optional)**: If you have access to a GPU runtime (Runtime > Change runtime type > GPU), it is recommended for faster training.

Enjoy learning about CNNs with PyTorch!
