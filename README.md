# Real-Time Multi-Class Object Detection with YOLOv8

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00)

An end-to-end multi-class object detection pipeline built with YOLOv8 and trained on the Pascal VOC dataset. This repository provides a streamlined workflow for dataset partitioning, model training, automated validation, and an interactive in-notebook inference interface powered by Google Colab widgets.

---

## Overview

This project implements a single-stage, anchor-free object detection system capable of identifying and localizing 20 distinct object categories across diverse visual scenes. Rather than relying on external web servers, the deployment layer is integrated directly into the notebook environment, enabling real-time threshold adjustments and side-by-side visual comparisons on custom user-uploaded photographs.

### Key Features
* **Automated Data Pipeline:** Downloads and structures the Pascal VOC 2007 and 2012 datasets into dedicated training, validation, and testing splits to prevent data leakage.
* **Modern Architecture:** Utilizes the YOLOv8s backbone and decoupled head for an optimal balance between localization accuracy and fast inference speeds.
* **Interactive Notebook UI:** Features an embedded IPyWidgets control panel that allows users to upload custom images and dynamically tune confidence and Non-Maximum Suppression IoU thresholds.
* **Real-World Testing Suite:** Supports batch evaluation on custom photographs with automated bounding box rendering and class-level detection summaries.

---

## Technical Specifications

| Component | Specification |
| :--- | :--- |
| **Task** | Multi-Class Object Detection |
| **Dataset** | Pascal VOC 2007 + 2012 |
| **Dataset Source** | [Pascal VOC Dataset Mirror](https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_11-May-2012.zip) |
| **Model Architecture** | YOLOv8s |
| **Input Resolution** | 640 x 640 RGB |
| **Optimizer** | AdamW |
| **Activation Functions** | SiLU and Sigmoid |
| **Loss Functions** | Binary Cross-Entropy, Complete IoU Loss, and Distribution Focal Loss |
| **Evaluation Metrics** | IoU, Precision, Recall, F1-Score, mAP@0.50, and mAP@0.50:0.95 |
| **Execution Environment** | Google Colab with NVIDIA GPU acceleration |
| **Interactive Interface** | IPyWidgets and Matplotlib |

---

## Methodology

### Dataset Structure
The pipeline processes 20 standard object classes: `aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, and `tvmonitor`. A custom YAML configuration separates the data into independent training, validation, and testing sets to ensure reliable generalization across unseen scenes.

### Training Pipeline
The network is optimized using AdamW with decoupled weight decay to mitigate overfitting across deep convolutional layers. Spatial and multi-scale feature extraction is handled by the CSPDarknet53 backbone paired with a Path Aggregation Network neck. Bounding box regression is guided by Complete IoU and Distribution Focal Loss, while multi-label class probabilities are computed independently via Binary Cross-Entropy.

---

## Getting Started

### 1. Installation
Open the notebook in Google Colab, select a GPU runtime, and install the required dependencies:

```bash
pip install ultralytics opencv-python matplotlib ipywidgets
```

### 2. Training and Validation
Run the training cell to generate the dataset configuration, download the Pascal VOC images, train the YOLOv8s model, and evaluate the best weights against the test split.

### 3. Interactive Inference on Custom Images
Execute the interactive application cell to test the trained model on your own photographs:
1. Click the file upload button to select custom images from your local device.
2. Adjust the **Confidence** and **NMS IoU** sliders directly inside the notebook cell.
3. View the original images alongside the predicted bounding boxes, class labels, and per-image inference timings.
