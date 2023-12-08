# ML_CNN_Models
This is a Computer Vision (CV) task in deep learning, using Convolutional  Neural Network (CNN) backbone models to address image classification problems

## Training the Model
The dataset must be placed in a folder called "Colorectal Cancer"
under the root of this project. Within that are sub-folders 
called MUS, NORM, etc.

# Task 1
### Description:
This repository contains three Python scripts for a machine learning project focusing on colorectal cancer classification and data visualization using the ResNet34 model in PyTorch.
Each python code serves a distinct role in the machine learning pipeline: training the model, testing and evaluating it, and applying t-SNE for data visualization.

### Dataset:
Dataset 1) Colorectal Cancer​

### Part1-Test-model.py (Model Evaluation)
This script evaluates a trained ResNet34 model on a colorectal cancer dataset.
Key Features:
Model Loading: Loads a pre-trained ResNet34 model.
Data Processing: Implements custom transformations for normalization.
Evaluation Metrics: Calculates confusion matrix, accuracy, and other classification metrics.
Visualization: Displays a confusion matrix for test data predictions.

### Part1-Train-model.py (Model Training)
Key Features:
Model Configuration: Sets up ResNet34 with custom class numbers.
Data Preparation: Loads and transforms the dataset.
Training Loop: Includes forward pass, backpropagation, and optimization steps.
Performance Monitoring: Tracks loss and accuracy on the validation set.

### Part1-t-SNE.py (Data Visualization)
Key Features:
Feature Extraction: Uses a ResNet34 encoder to extract features from the dataset.
t-SNE Application: Reduces the dimensionality of the extracted features.
Plotting: Visualizes the data points in a 2D scatter plot, color-coded by class

### Requirements
Python 3, PyTorch, torchvision, scikit-learn, Matplotlib

# Task 2:
Task2.ipnyb

### Description:
This machine learning program is designed to perform feature extraction and classification tasks using deep learning models. It utilizes PyTorch for model building and training, leveraging the ResNet34 architecture for feature extraction from Dataset 1. The program is capable of processing images, extracting features, and classifying them into different categories.

### Features:
Model Building: Use of ResNet34 for feature extraction.
Data Preprocessing: Implements custom transformations for data normalization and augmentation.
Feature Extraction: Extracts features from datasets using pre-trained and custom models.
t-SNE Visualization: Performs t-Distributed Stochastic Neighbor Embedding (t-SNE) for high-dimensional data visualization.
Classification with SVM: Utilizes Support Vector Machine (SVM) for the classification of extracted features.

### Functions:
extract_features: Extracts features and labels from the dataset.
ImageNetModel: Creates a model pre-trained on ImageNet.
find_perplexity: Finds and plots KL divergence for a range of perplexity values.
perform_tsne: Performs t-SNE on extracted features and visualizes the results.

### Datasets:
Dataset 1) Colorectal Cancer​
Dataset 2) Prostate Cancer​
Dataset 3) animal faces 

### Requirements:
Python 3, PyTorch, torchvision, scikit-learn, NumPy, Matplotlib, Plotly
