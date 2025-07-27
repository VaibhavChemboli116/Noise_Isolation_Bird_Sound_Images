# Noise Isolation in Bird Sound Spectrograms using AtrousSegNet

## Overview

This project implements a deep learning model for image segmentation to isolate bird sounds from background noise in spectrogram images. The model, named **AtrousSegNet**, is built from scratch using PyTorch and leverages atrous (dilated) convolutions to capture multi-scale contextual information, which is crucial for identifying the fine details in spectrograms. The primary goal is to generate an accurate binary mask that highlights the regions corresponding to bird sounds.

## Dataset

The model is trained on a specialized dataset of bird sound spectrograms. The dataset is split into training, validation, and test sets. Each sample consists of:
- An RGB spectrogram image.
- A corresponding binary ground truth mask where the bird sound is segmented.

The images and masks are resized to **256x256 pixels** for training and evaluation.

## Model Architecture: AtrousSegNet

AtrousSegNet is a custom encoder-decoder network designed for semantic segmentation.

- **Encoder**: The encoder uses a series of `AtrousConvBlock` modules with increasing dilation rates (1, 2, 4). This allows the network to learn features at different scales without losing spatial resolution, which is often a side effect of pooling layers. Max pooling is used to downsample the feature maps between blocks.

- **Decoder**: The decoder upsamples the feature maps using `ConvTranspose2d` layers and combines them with skip connections from the corresponding encoder blocks. This helps in recovering spatial details lost during encoding. The final output is a single-channel mask.

## Methodology

1. **Data Loading**: A custom `BirdSoundDataset` class is created in PyTorch to load and preprocess the images and masks from the respective directories.

2. **Data Transformation**: The input images are converted to tensors and normalized. The masks are converted to single-channel float tensors.

3. **Training**:
   - The model is trained for **128 epochs** using the **Adam** optimizer and **Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss)**.
   - The **Intersection over Union (IoU)** metric is used to evaluate the model's performance on the validation set after each epoch.
   - The model state with the best validation IoU is saved for later use.

4. **Evaluation**: The trained model is evaluated on the test set to measure its final performance, achieving a **Test Set IoU of 65.33%**, which is more than the pre-trained model **DeepLabV3**.

## Results

The model successfully learns to segment bird sounds from the spectrograms. The final **validation IoU** reached **66.28%**, and the **test set IoU** was **65.33%**, indicating that the model generalizes well to unseen data.

Here is an example of the model's prediction on a test image:

<img width="468" height="75" alt="image" src="https://github.com/user-attachments/assets/fd4c9f54-80ac-45ea-bae3-9de95aad6730" />

## Report

A detailed report on this project, including the methodology and results, has been published on ResearchGate. You can access the paper via the following link:

A detailed report on this project, including the methodology and results, has been published on ResearchGate:  
👉 [**Noise Isolation in Bird Sound Images Using AtrousSegNet with Multi-Scale Contextual Learning**](https://www.researchgate.net/publication/387180246_Noise_Isolation_in_Bird_Sound_Images_Using_AtrousSegNet_with_Multi-Scale_Contextual_Learning)
