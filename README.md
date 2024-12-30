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

Link best_model.keras: 


<https://tinyurl.com/2wju8t33>


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

The script will generate a console output such as the following (shortened):

```bash
Event discarded.
Event discarded.
Event image saved as '000000_PCe.png'.
Event discarded.
Event image saved as '000001_FCmu.png'.
Event image saved as '000002_FCmu.png'.
Event discarded.
Event image saved as '000003_FCe.png'.
Event discarded.
Event image saved as '000004_PCe.png'.
[...]
Event image saved as '000999_FCe.png'.
```

### Train the Model

1. Ensure **train_model.py** points to the directory containing the generated images (default: `event_display_new`).  
2. Run the script:

```bash
python train_model.py
```

3. A ResNet50-based classifier will be trained.  
4. A model file (default: `best_model.keras`) will be saved according to the training script’s configuration and callbacks.

During execution the script will produce a text and graphical output such as the following:

```bash
2024-12-29 18:31:15.348303: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
TensorFlow version: 2.16.2
Available GPU devices: []
Image counts per class:
   FCe: 238 images
   FCmu: 256 images
   PCe: 273 images
   PCmu: 233 images
```

![](train1.png)


```bash
Found 703 images belonging to 4 classes.
Found 297 images belonging to 4 classes.
Class indices: {'FCe': 0, 'FCmu': 1, 'PCe': 2, 'PCmu': 3}
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (None, 224, 224,  │          0 │ -                 │
│ (InputLayer)        │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1_pad           │ (None, 230, 230,  │          0 │ input_layer[0][0] │
│ (ZeroPadding2D)     │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1_conv (Conv2D) │ (None, 112, 112,  │      9,472 │ conv1_pad[0][0]   │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1_bn            │ (None, 112, 112,  │        256 │ conv1_conv[0][0]  │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1_relu          │ (None, 112, 112,  │          0 │ conv1_bn[0][0]    │
│ (Activation)        │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ pool1_pad           │ (None, 114, 114,  │          0 │ conv1_relu[0][0]  │
│ (ZeroPadding2D)     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ pool1_pool          │ (None, 56, 56,    │          0 │ pool1_pad[0][0]   │
│ (MaxPooling2D)      │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_1_conv │ (None, 56, 56,    │      4,160 │ pool1_pool[0][0]  │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_1_bn   │ (None, 56, 56,    │        256 │ conv2_block1_1_c… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_1_relu │ (None, 56, 56,    │          0 │ conv2_block1_1_b… │
│ (Activation)        │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_2_conv │ (None, 56, 56,    │     36,928 │ conv2_block1_1_r… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_2_bn   │ (None, 56, 56,    │        256 │ conv2_block1_2_c… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_2_relu │ (None, 56, 56,    │          0 │ conv2_block1_2_b… │
│ (Activation)        │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_0_conv │ (None, 56, 56,    │     16,640 │ pool1_pool[0][0]  │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_3_conv │ (None, 56, 56,    │     16,640 │ conv2_block1_2_r… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_0_bn   │ (None, 56, 56,    │      1,024 │ conv2_block1_0_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_3_bn   │ (None, 56, 56,    │      1,024 │ conv2_block1_3_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_add    │ (None, 56, 56,    │          0 │ conv2_block1_0_b… │
│ (Add)               │ 256)              │            │ conv2_block1_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block1_out    │ (None, 56, 56,    │          0 │ conv2_block1_add… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_1_conv │ (None, 56, 56,    │     16,448 │ conv2_block1_out… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_1_bn   │ (None, 56, 56,    │        256 │ conv2_block2_1_c… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_1_relu │ (None, 56, 56,    │          0 │ conv2_block2_1_b… │
│ (Activation)        │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_2_conv │ (None, 56, 56,    │     36,928 │ conv2_block2_1_r… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_2_bn   │ (None, 56, 56,    │        256 │ conv2_block2_2_c… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_2_relu │ (None, 56, 56,    │          0 │ conv2_block2_2_b… │
│ (Activation)        │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_3_conv │ (None, 56, 56,    │     16,640 │ conv2_block2_2_r… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_3_bn   │ (None, 56, 56,    │      1,024 │ conv2_block2_3_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_add    │ (None, 56, 56,    │          0 │ conv2_block1_out… │
│ (Add)               │ 256)              │            │ conv2_block2_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block2_out    │ (None, 56, 56,    │          0 │ conv2_block2_add… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_1_conv │ (None, 56, 56,    │     16,448 │ conv2_block2_out… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_1_bn   │ (None, 56, 56,    │        256 │ conv2_block3_1_c… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_1_relu │ (None, 56, 56,    │          0 │ conv2_block3_1_b… │
│ (Activation)        │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_2_conv │ (None, 56, 56,    │     36,928 │ conv2_block3_1_r… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_2_bn   │ (None, 56, 56,    │        256 │ conv2_block3_2_c… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_2_relu │ (None, 56, 56,    │          0 │ conv2_block3_2_b… │
│ (Activation)        │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_3_conv │ (None, 56, 56,    │     16,640 │ conv2_block3_2_r… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_3_bn   │ (None, 56, 56,    │      1,024 │ conv2_block3_3_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_add    │ (None, 56, 56,    │          0 │ conv2_block2_out… │
│ (Add)               │ 256)              │            │ conv2_block3_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2_block3_out    │ (None, 56, 56,    │          0 │ conv2_block3_add… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_1_conv │ (None, 28, 28,    │     32,896 │ conv2_block3_out… │
│ (Conv2D)            │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_1_bn   │ (None, 28, 28,    │        512 │ conv3_block1_1_c… │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_1_relu │ (None, 28, 28,    │          0 │ conv3_block1_1_b… │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_2_conv │ (None, 28, 28,    │    147,584 │ conv3_block1_1_r… │
│ (Conv2D)            │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_2_bn   │ (None, 28, 28,    │        512 │ conv3_block1_2_c… │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_2_relu │ (None, 28, 28,    │          0 │ conv3_block1_2_b… │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_0_conv │ (None, 28, 28,    │    131,584 │ conv2_block3_out… │
│ (Conv2D)            │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_3_conv │ (None, 28, 28,    │     66,048 │ conv3_block1_2_r… │
│ (Conv2D)            │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_0_bn   │ (None, 28, 28,    │      2,048 │ conv3_block1_0_c… │
│ (BatchNormalizatio… │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_3_bn   │ (None, 28, 28,    │      2,048 │ conv3_block1_3_c… │
│ (BatchNormalizatio… │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_add    │ (None, 28, 28,    │          0 │ conv3_block1_0_b… │
│ (Add)               │ 512)              │            │ conv3_block1_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block1_out    │ (None, 28, 28,    │          0 │ conv3_block1_add… │
│ (Activation)        │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_1_conv │ (None, 28, 28,    │     65,664 │ conv3_block1_out… │
│ (Conv2D)            │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_1_bn   │ (None, 28, 28,    │        512 │ conv3_block2_1_c… │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_1_relu │ (None, 28, 28,    │          0 │ conv3_block2_1_b… │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_2_conv │ (None, 28, 28,    │    147,584 │ conv3_block2_1_r… │
│ (Conv2D)            │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_2_bn   │ (None, 28, 28,    │        512 │ conv3_block2_2_c… │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_2_relu │ (None, 28, 28,    │          0 │ conv3_block2_2_b… │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_3_conv │ (None, 28, 28,    │     66,048 │ conv3_block2_2_r… │
│ (Conv2D)            │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_3_bn   │ (None, 28, 28,    │      2,048 │ conv3_block2_3_c… │
│ (BatchNormalizatio… │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_add    │ (None, 28, 28,    │          0 │ conv3_block1_out… │
│ (Add)               │ 512)              │            │ conv3_block2_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block2_out    │ (None, 28, 28,    │          0 │ conv3_block2_add… │
│ (Activation)        │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_1_conv │ (None, 28, 28,    │     65,664 │ conv3_block2_out… │
│ (Conv2D)            │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_1_bn   │ (None, 28, 28,    │        512 │ conv3_block3_1_c… │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_1_relu │ (None, 28, 28,    │          0 │ conv3_block3_1_b… │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_2_conv │ (None, 28, 28,    │    147,584 │ conv3_block3_1_r… │
│ (Conv2D)            │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_2_bn   │ (None, 28, 28,    │        512 │ conv3_block3_2_c… │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_2_relu │ (None, 28, 28,    │          0 │ conv3_block3_2_b… │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_3_conv │ (None, 28, 28,    │     66,048 │ conv3_block3_2_r… │
│ (Conv2D)            │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_3_bn   │ (None, 28, 28,    │      2,048 │ conv3_block3_3_c… │
│ (BatchNormalizatio… │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_add    │ (None, 28, 28,    │          0 │ conv3_block2_out… │
│ (Add)               │ 512)              │            │ conv3_block3_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block3_out    │ (None, 28, 28,    │          0 │ conv3_block3_add… │
│ (Activation)        │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_1_conv │ (None, 28, 28,    │     65,664 │ conv3_block3_out… │
│ (Conv2D)            │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_1_bn   │ (None, 28, 28,    │        512 │ conv3_block4_1_c… │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_1_relu │ (None, 28, 28,    │          0 │ conv3_block4_1_b… │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_2_conv │ (None, 28, 28,    │    147,584 │ conv3_block4_1_r… │
│ (Conv2D)            │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_2_bn   │ (None, 28, 28,    │        512 │ conv3_block4_2_c… │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_2_relu │ (None, 28, 28,    │          0 │ conv3_block4_2_b… │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_3_conv │ (None, 28, 28,    │     66,048 │ conv3_block4_2_r… │
│ (Conv2D)            │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_3_bn   │ (None, 28, 28,    │      2,048 │ conv3_block4_3_c… │
│ (BatchNormalizatio… │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_add    │ (None, 28, 28,    │          0 │ conv3_block3_out… │
│ (Add)               │ 512)              │            │ conv3_block4_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3_block4_out    │ (None, 28, 28,    │          0 │ conv3_block4_add… │
│ (Activation)        │ 512)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_1_conv │ (None, 14, 14,    │    131,328 │ conv3_block4_out… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_1_bn   │ (None, 14, 14,    │      1,024 │ conv4_block1_1_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_1_relu │ (None, 14, 14,    │          0 │ conv4_block1_1_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_2_conv │ (None, 14, 14,    │    590,080 │ conv4_block1_1_r… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_2_bn   │ (None, 14, 14,    │      1,024 │ conv4_block1_2_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_2_relu │ (None, 14, 14,    │          0 │ conv4_block1_2_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_0_conv │ (None, 14, 14,    │    525,312 │ conv3_block4_out… │
│ (Conv2D)            │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_3_conv │ (None, 14, 14,    │    263,168 │ conv4_block1_2_r… │
│ (Conv2D)            │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_0_bn   │ (None, 14, 14,    │      4,096 │ conv4_block1_0_c… │
│ (BatchNormalizatio… │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_3_bn   │ (None, 14, 14,    │      4,096 │ conv4_block1_3_c… │
│ (BatchNormalizatio… │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_add    │ (None, 14, 14,    │          0 │ conv4_block1_0_b… │
│ (Add)               │ 1024)             │            │ conv4_block1_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block1_out    │ (None, 14, 14,    │          0 │ conv4_block1_add… │
│ (Activation)        │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_1_conv │ (None, 14, 14,    │    262,400 │ conv4_block1_out… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_1_bn   │ (None, 14, 14,    │      1,024 │ conv4_block2_1_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_1_relu │ (None, 14, 14,    │          0 │ conv4_block2_1_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_2_conv │ (None, 14, 14,    │    590,080 │ conv4_block2_1_r… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_2_bn   │ (None, 14, 14,    │      1,024 │ conv4_block2_2_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_2_relu │ (None, 14, 14,    │          0 │ conv4_block2_2_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_3_conv │ (None, 14, 14,    │    263,168 │ conv4_block2_2_r… │
│ (Conv2D)            │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_3_bn   │ (None, 14, 14,    │      4,096 │ conv4_block2_3_c… │
│ (BatchNormalizatio… │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_add    │ (None, 14, 14,    │          0 │ conv4_block1_out… │
│ (Add)               │ 1024)             │            │ conv4_block2_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block2_out    │ (None, 14, 14,    │          0 │ conv4_block2_add… │
│ (Activation)        │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_1_conv │ (None, 14, 14,    │    262,400 │ conv4_block2_out… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_1_bn   │ (None, 14, 14,    │      1,024 │ conv4_block3_1_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_1_relu │ (None, 14, 14,    │          0 │ conv4_block3_1_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_2_conv │ (None, 14, 14,    │    590,080 │ conv4_block3_1_r… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_2_bn   │ (None, 14, 14,    │      1,024 │ conv4_block3_2_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_2_relu │ (None, 14, 14,    │          0 │ conv4_block3_2_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_3_conv │ (None, 14, 14,    │    263,168 │ conv4_block3_2_r… │
│ (Conv2D)            │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_3_bn   │ (None, 14, 14,    │      4,096 │ conv4_block3_3_c… │
│ (BatchNormalizatio… │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_add    │ (None, 14, 14,    │          0 │ conv4_block2_out… │
│ (Add)               │ 1024)             │            │ conv4_block3_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block3_out    │ (None, 14, 14,    │          0 │ conv4_block3_add… │
│ (Activation)        │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_1_conv │ (None, 14, 14,    │    262,400 │ conv4_block3_out… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_1_bn   │ (None, 14, 14,    │      1,024 │ conv4_block4_1_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_1_relu │ (None, 14, 14,    │          0 │ conv4_block4_1_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_2_conv │ (None, 14, 14,    │    590,080 │ conv4_block4_1_r… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_2_bn   │ (None, 14, 14,    │      1,024 │ conv4_block4_2_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_2_relu │ (None, 14, 14,    │          0 │ conv4_block4_2_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_3_conv │ (None, 14, 14,    │    263,168 │ conv4_block4_2_r… │
│ (Conv2D)            │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_3_bn   │ (None, 14, 14,    │      4,096 │ conv4_block4_3_c… │
│ (BatchNormalizatio… │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_add    │ (None, 14, 14,    │          0 │ conv4_block3_out… │
│ (Add)               │ 1024)             │            │ conv4_block4_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block4_out    │ (None, 14, 14,    │          0 │ conv4_block4_add… │
│ (Activation)        │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_1_conv │ (None, 14, 14,    │    262,400 │ conv4_block4_out… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_1_bn   │ (None, 14, 14,    │      1,024 │ conv4_block5_1_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_1_relu │ (None, 14, 14,    │          0 │ conv4_block5_1_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_2_conv │ (None, 14, 14,    │    590,080 │ conv4_block5_1_r… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_2_bn   │ (None, 14, 14,    │      1,024 │ conv4_block5_2_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_2_relu │ (None, 14, 14,    │          0 │ conv4_block5_2_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_3_conv │ (None, 14, 14,    │    263,168 │ conv4_block5_2_r… │
│ (Conv2D)            │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_3_bn   │ (None, 14, 14,    │      4,096 │ conv4_block5_3_c… │
│ (BatchNormalizatio… │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_add    │ (None, 14, 14,    │          0 │ conv4_block4_out… │
│ (Add)               │ 1024)             │            │ conv4_block5_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block5_out    │ (None, 14, 14,    │          0 │ conv4_block5_add… │
│ (Activation)        │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_1_conv │ (None, 14, 14,    │    262,400 │ conv4_block5_out… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_1_bn   │ (None, 14, 14,    │      1,024 │ conv4_block6_1_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_1_relu │ (None, 14, 14,    │          0 │ conv4_block6_1_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_2_conv │ (None, 14, 14,    │    590,080 │ conv4_block6_1_r… │
│ (Conv2D)            │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_2_bn   │ (None, 14, 14,    │      1,024 │ conv4_block6_2_c… │
│ (BatchNormalizatio… │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_2_relu │ (None, 14, 14,    │          0 │ conv4_block6_2_b… │
│ (Activation)        │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_3_conv │ (None, 14, 14,    │    263,168 │ conv4_block6_2_r… │
│ (Conv2D)            │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_3_bn   │ (None, 14, 14,    │      4,096 │ conv4_block6_3_c… │
│ (BatchNormalizatio… │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_add    │ (None, 14, 14,    │          0 │ conv4_block5_out… │
│ (Add)               │ 1024)             │            │ conv4_block6_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4_block6_out    │ (None, 14, 14,    │          0 │ conv4_block6_add… │
│ (Activation)        │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_1_conv │ (None, 7, 7, 512) │    524,800 │ conv4_block6_out… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_1_bn   │ (None, 7, 7, 512) │      2,048 │ conv5_block1_1_c… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_1_relu │ (None, 7, 7, 512) │          0 │ conv5_block1_1_b… │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_2_conv │ (None, 7, 7, 512) │  2,359,808 │ conv5_block1_1_r… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_2_bn   │ (None, 7, 7, 512) │      2,048 │ conv5_block1_2_c… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_2_relu │ (None, 7, 7, 512) │          0 │ conv5_block1_2_b… │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_0_conv │ (None, 7, 7,      │  2,099,200 │ conv4_block6_out… │
│ (Conv2D)            │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_3_conv │ (None, 7, 7,      │  1,050,624 │ conv5_block1_2_r… │
│ (Conv2D)            │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_0_bn   │ (None, 7, 7,      │      8,192 │ conv5_block1_0_c… │
│ (BatchNormalizatio… │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_3_bn   │ (None, 7, 7,      │      8,192 │ conv5_block1_3_c… │
│ (BatchNormalizatio… │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_add    │ (None, 7, 7,      │          0 │ conv5_block1_0_b… │
│ (Add)               │ 2048)             │            │ conv5_block1_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block1_out    │ (None, 7, 7,      │          0 │ conv5_block1_add… │
│ (Activation)        │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_1_conv │ (None, 7, 7, 512) │  1,049,088 │ conv5_block1_out… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_1_bn   │ (None, 7, 7, 512) │      2,048 │ conv5_block2_1_c… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_1_relu │ (None, 7, 7, 512) │          0 │ conv5_block2_1_b… │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_2_conv │ (None, 7, 7, 512) │  2,359,808 │ conv5_block2_1_r… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_2_bn   │ (None, 7, 7, 512) │      2,048 │ conv5_block2_2_c… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_2_relu │ (None, 7, 7, 512) │          0 │ conv5_block2_2_b… │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_3_conv │ (None, 7, 7,      │  1,050,624 │ conv5_block2_2_r… │
│ (Conv2D)            │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_3_bn   │ (None, 7, 7,      │      8,192 │ conv5_block2_3_c… │
│ (BatchNormalizatio… │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_add    │ (None, 7, 7,      │          0 │ conv5_block1_out… │
│ (Add)               │ 2048)             │            │ conv5_block2_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block2_out    │ (None, 7, 7,      │          0 │ conv5_block2_add… │
│ (Activation)        │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_1_conv │ (None, 7, 7, 512) │  1,049,088 │ conv5_block2_out… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_1_bn   │ (None, 7, 7, 512) │      2,048 │ conv5_block3_1_c… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_1_relu │ (None, 7, 7, 512) │          0 │ conv5_block3_1_b… │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_2_conv │ (None, 7, 7, 512) │  2,359,808 │ conv5_block3_1_r… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_2_bn   │ (None, 7, 7, 512) │      2,048 │ conv5_block3_2_c… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_2_relu │ (None, 7, 7, 512) │          0 │ conv5_block3_2_b… │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_3_conv │ (None, 7, 7,      │  1,050,624 │ conv5_block3_2_r… │
│ (Conv2D)            │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_3_bn   │ (None, 7, 7,      │      8,192 │ conv5_block3_3_c… │
│ (BatchNormalizatio… │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_add    │ (None, 7, 7,      │          0 │ conv5_block2_out… │
│ (Add)               │ 2048)             │            │ conv5_block3_3_b… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5_block3_out    │ (None, 7, 7,      │          0 │ conv5_block3_add… │
│ (Activation)        │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (None, 2048)      │          0 │ conv5_block3_out… │
│ (GlobalAveragePool… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (Dense)       │ (None, 256)       │    524,544 │ global_average_p… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout (Dropout)   │ (None, 256)       │          0 │ dense[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (Dense)     │ (None, 4)         │      1,028 │ dropout[0][0]     │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 24,113,284 (91.98 MB)
 Trainable params: 525,572 (2.00 MB)
 Non-trainable params: 23,587,712 (89.98 MB)
Epoch 1/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 75s 2s/step - accuracy: 0.3465 - loss: 1.4604 - val_accuracy: 0.7374 - val_loss: 0.9838
Epoch 2/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.5185 - loss: 1.1174 - val_accuracy: 0.8889 - val_loss: 0.7658
Epoch 3/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.6559 - loss: 0.8363 - val_accuracy: 0.9024 - val_loss: 0.6381
Epoch 4/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.7458 - loss: 0.7085 - val_accuracy: 0.9024 - val_loss: 0.5471
Epoch 5/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.8115 - loss: 0.6022 - val_accuracy: 0.9125 - val_loss: 0.4878
Epoch 6/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 66s 2s/step - accuracy: 0.8218 - loss: 0.5389 - val_accuracy: 0.9226 - val_loss: 0.4375
Epoch 7/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.8971 - loss: 0.4511 - val_accuracy: 0.9125 - val_loss: 0.3931
Epoch 8/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.8920 - loss: 0.4158 - val_accuracy: 0.9259 - val_loss: 0.3602
Epoch 9/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9039 - loss: 0.3740 - val_accuracy: 0.9293 - val_loss: 0.3277
Epoch 10/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9034 - loss: 0.3769 - val_accuracy: 0.9529 - val_loss: 0.2881
Epoch 11/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9140 - loss: 0.3457 - val_accuracy: 0.9360 - val_loss: 0.2786
Epoch 12/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.8916 - loss: 0.3464 - val_accuracy: 0.9394 - val_loss: 0.2651
Epoch 13/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9200 - loss: 0.3248 - val_accuracy: 0.9293 - val_loss: 0.2508
Epoch 14/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9170 - loss: 0.3009 - val_accuracy: 0.9428 - val_loss: 0.2445
Epoch 15/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9339 - loss: 0.2798 - val_accuracy: 0.9495 - val_loss: 0.2142
Epoch 16/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9151 - loss: 0.2610 - val_accuracy: 0.9495 - val_loss: 0.2128
Epoch 17/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9355 - loss: 0.2439 - val_accuracy: 0.9461 - val_loss: 0.1947
Epoch 18/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9331 - loss: 0.2289 - val_accuracy: 0.9562 - val_loss: 0.1912
Epoch 19/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9396 - loss: 0.2221 - val_accuracy: 0.9596 - val_loss: 0.1776
Epoch 20/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 66s 2s/step - accuracy: 0.9523 - loss: 0.2204 - val_accuracy: 0.9327 - val_loss: 0.1978
Epoch 21/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9394 - loss: 0.2126 - val_accuracy: 0.9529 - val_loss: 0.1690
Epoch 22/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9235 - loss: 0.2335 - val_accuracy: 0.9562 - val_loss: 0.1689
Epoch 23/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9339 - loss: 0.2205 - val_accuracy: 0.9562 - val_loss: 0.1612
Epoch 24/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 67s 2s/step - accuracy: 0.9409 - loss: 0.2181 - val_accuracy: 0.9663 - val_loss: 0.1452
Epoch 25/25
44/44 ━━━━━━━━━━━━━━━━━━━━ 66s 2s/step - accuracy: 0.9674 - loss: 0.1505 - val_accuracy: 0.9529 - val_loss: 0.1623
```

### Evaluate the Model

1. Once training is complete, run:

```bash
python evaluate_model.py
```

2. Ensure that **evaluate_model.py** references the correct path for the test set (could be within the same directory if a validation split is used, or a separate folder for a truly unseen test).  
3. The script outputs confusion matrices, classification reports, and plots ROC curves (or one-vs-rest curves for multi-class scenarios).  
4. Check the console output and any generated figures to interpret the classifier’s performance.

During execution the script will produce a text and graphical output such as the following:

```bash

2024-12-29 22:46:25.342048: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 100ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 104ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 98ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 107ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 100ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 105ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 104ms/step
[...]
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 104ms/step

Confusion Matrix:
 [[195   0  21   0]
 [  0 261   0   0]
 [ 12   0 240   0]
 [  0   1   0 270]]

Classification Report:
               precision    recall  f1-score   support

         FCe       0.94      0.90      0.92       216
        FCmu       1.00      1.00      1.00       261
         PCe       0.92      0.95      0.94       252
        PCmu       1.00      1.00      1.00       271

    accuracy                           0.97      1000
   macro avg       0.96      0.96      0.96      1000
weighted avg       0.97      0.97      0.97      1000

F1 Score (Macro): 0.9635
F1 Score (Weighted): 0.9659
Confusion Matrix, without normalization
Normalized Confusion Matrix
lorenzo@MacBook Pro sc_GitHub % 
```
![](evaluate1.png)
![](evaluate2.png)
![](evaluate3.png)
![](evaluate4.png)
![](evaluate5.png)
![](evaluate6.png)

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