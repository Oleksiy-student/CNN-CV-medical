# ML_CNN_Models
This is a Computer Vision (CV) task in deep learning, using Convolutional  Neural Network (CNN) backbone models to address image classification problems

## Training the Model
The dataset must be placed in a folder called "Colorectal Cancer"
under the root of this project. Within that are sub-folders 
called MUS, NORM, etc.
################################################################
#Task2.ipnyb

Description:
This machine learning program is designed to perform feature extraction and classification tasks using deep learning models. It utilizes PyTorch for model building and training, leveraging the ResNet34 architecture for feature extraction. The program is capable of processing images, extracting features, and classifying them into different categories.

Features:
Model Building: Use of ResNet34 for feature extraction.
Data Preprocessing: Implements custom transformations for data normalization and augmentation.
Feature Extraction: Extracts features from datasets using pre-trained and custom models.
t-SNE Visualization: Performs t-Distributed Stochastic Neighbor Embedding (t-SNE) for high-dimensional data visualization.
Classification with SVM: Utilizes Support Vector Machine (SVM) for the classification of extracted features.

Functions:
extract_features: Extracts features and labels from the dataset.
ImageNetModel: Creates a model pre-trained on ImageNet.
find_perplexity: Finds and plots KL divergence for a range of perplexity values.
perform_tsne: Performs t-SNE on extracted features and visualizes the results.

Requirements:
Python 3, PyTorch, torchvision, scikit-learn, NumPy, Matplotlib, Plotly
################################################################