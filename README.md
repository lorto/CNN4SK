# CNN4SK: A Deep Learning Approach for Cherenkov Event Classification

![](sk.jpg)

### Table of Contents

1. [Introduction](#introduction)  
2. [Project Structure](#project-structure)
3. [Scripts Overview](#scripts-overview)  
   - [generate_events.py](#generate_eventspy)  
   - [train_model.py](#train_modelpy)  
   - [evaluate_model.py](#evaluate_modelpy)  
4. [Usage Instructions](#usage-instructions)  
   - [Generate the Dataset](#generate-the-dataset)  
   - [Train the Model](#train-the-model)  
   - [Evaluate the Model](#evaluate-the-model)
5. [Key Libraries and Requirements](#key-libraries-and-requirements)

## Introduction

This repository contains an end-to-end workflow for simulating Cherenkov events in a water-based detector (inspired by Super-Kamiokande) and classifying these events using a Deep Learning model (ResNet50). The project is intended to illustrate how synthetic data generation, training, and evaluation can be integrated in a physics-inspired context.

For physics details please refer to physics(intro).pdf.

This project:

- **Generates** synthetic 2D images that approximate Cherenkov rings for different particle types (e or µ) and topologies (Fully Contained or Partially Contained).  
- **Trains** a CNN-based classifier (ResNet50 backbone) to differentiate the events.  
- **Evaluates** the trained model on a test set, providing classification metrics and visualization (Confusion Matrix, ROC curves).

## Project Structure

A possible directory structure is shown below:

```bash
my_cherenkov_project/ 
├── generate_events.py # Script for simulating Cherenkov images
├── train_model.py # Script for training ResNet50-based classifier 
├── evaluate_model.py # Script for evaluating trained model 
├── event_display_new/ # Default output folder for generated images 
│ 	├── FCe/ # Fully Contained, electron-like events 
│ 	│ ├── 000000_FCe.png # Synthetic event examples 
│	│ ├── 000001_FCe.png 
│ 	│ └── ... 
│	├── FCmu/ # Fully Contained, muon-like events 
│ 	├── PCe/ # Partially Contained, electron-like events 
│ 	└── PCmu/ # Partially Contained, muon-like events ├── best_model.keras # Saved model weights (if training is successful) 
└── README.md # Current documentation file
```

- **generate_events.py**  
  Produces synthetic Cherenkov event images and saves them to subfolders according to their class (FCe, FCmu, PCe, PCmu).

- **train_model.py**  
  Loads the generated dataset, sets up the ResNet50 model with custom layers, and trains the classifier.

- **evaluate_model.py**  
  Loads the trained model for inference and produces performance metrics, including confusion matrices and ROC curves.

- **event_display_new/**  
  A directory of generated event images, subdivided by topology (FC or PC) and particle type (e or mu).

- **best_model.keras**  
  The saved model weights (if the training script was configured to save them under this name).

- **README.md**  
  The primary documentation file explaining the project’s objectives, setup, and usage instructions.

## Scripts Overview

### ```generate_events.py``` :warning:

**Role**: Synthetic Data Generation  
1. **Parameter Randomization**: Chooses whether an event is e-like or µ-like, and whether it is fully or partially contained. Random angles (azimuthal, polar) and distances are also sampled.  
2. **Cherenkov Cone Intersection**: Computes where the Cherenkov cone intersects the plane, resulting in elliptical shapes.  
3. **Noise Application**: Adds Gaussian noise to the ellipse points to mimic scattering. e-like events receive more diffuse patterns.  
4. **Output**: Saves each valid event as a PNG image into one of the four subfolders (FCe, FCmu, PCe, PCmu).

### train_model.py

**Role**: Model Training  
1. **Data Loading**: Reads images from the event_display_new/ directory using a Keras ImageDataGenerator for data augmentation.  
2. **Model Construction**:  
   - Loads a ResNet50 (pre-trained on ImageNet).  
   - Freezes the base layers.  
   - Adds a GlobalAveragePooling2D layer plus dense layers for final classification into four classes (FCe, FCmu, PCe, PCmu) or two classes depending on the classification approach.  
3. **Training Loop**:  
   - Optimizes the network via Adam or another suitable optimizer.  
   - Tracks validation performance; saves the best model weights to best_model.keras (if configured).  

### evaluate_model.py

**Role**: Model Evaluation  
1. **Model Loading**: Loads best_model.keras or the specified checkpoint.  
2. **Inference**: Runs predictions on a test dataset or a directory of images.  
3. **Performance Metrics**:  
   - Confusion Matrix (to visualize class-specific errors).  
   - Classification Report (precision, recall, F1).  
   - ROC Curves (one-vs-rest or one-vs-one).  
4. **Visualization**: Plots relevant figures (e.g., confusion matrix, normalized confusion matrix, per-class ROC curves).

## Usage Instructions

### Generate the Dataset

1. Modify parameters in **generate_events.py** if needed (e.g., number of images, image size, noise levels).  
2. Run the script in a terminal:
```bash
python generate_events.py
```
3. Check that a new directory (e.g., `event_display_new`) has been created with four subfolders: `FCe`, `FCmu`, `PCe` and `PCmu`.  

### Train the Model

1. Ensure **train_model.py** points to the directory containing the generated images (default: `event_display_new`).  
2. Run the script:

```bash
python train_model.py
```

3. A ResNet50-based classifier will be trained.  
4. A model file (default: `best_model.keras`) will be saved according to the training script’s configuration and callbacks.

### Evaluate the Model

1. Once training is complete, run:

```bash
python evaluate_model.py
```

2. Ensure that **evaluate_model.py** references the correct path for the test set (could be within the same directory if a validation split is used, or a separate folder for a truly unseen test).  
3. The script outputs confusion matrices, classification reports, and plots ROC curves (or one-vs-rest curves for multi-class scenarios).  
4. Check the console output and any generated figures to interpret the classifier’s performance.

## Key Libraries and Requirements

- **Python 3.7+** (recommended)  
- **tensorflow** or **keras** (for deep learning functionality)  
- **numpy** (array operations and numerical computations)  
- **matplotlib** (visualizations for confusion matrix, ROC curves)  
- **Pillow (PIL)** (image reading/writing)  
- **scikit-learn** (classification metrics, confusion matrix, ROC utilities)

Installation example (using pip):

```bash
pip install tensorflow numpy matplotlib pillow scikit-learn
```