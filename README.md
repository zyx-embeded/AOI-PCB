# PCB Automated Optical Inspection (AOI)

An Automated Optical Inspection (AOI) system for detecting and analyzing common Printed Circuit Board (PCB) defects using a combination of **deep learning** and **rule-based image processing**.

The project is designed to explore how modern object detection can be combined with traditional AOI inspection techniques commonly used in industrial PCB inspection systems.

## Overview

The system combines:

* **YOLO-based defect detection** for identifying PCB defects
* **Image registration** for aligning PCB images with a reference
* **Rule-based image processing** for geometric and connectivity-based inspection
* **Connectivity analysis** for detecting defects in PCB traces and conductive regions

The goal is to build a practical PCB inspection pipeline that combines the flexibility of deep learning with the deterministic behavior of traditional AOI algorithms.

## Supported Defects

The current YOLO-based detection model is trained to detect:

* Open Circuit
* Short Circuit
* Spur
* Mouse Bite
* Missing Hole

Additional rule-based inspection algorithms are being developed to improve detection accuracy and provide more deterministic inspection results.

## Inspection Pipeline

The planned inspection pipeline is structured as:

```text
PCB Image
    │
    ▼
Image Preprocessing
    │
    ▼
Image Registration
    │
    ▼
ROI / PCB Region Extraction
    │
    ├───────────────┐
    ▼               ▼
YOLO Detection   Rule-Based AOI
    │               │
    │               ├── Connectivity Analysis
    │               ├── Open Circuit
    │               ├── Short Circuit
    │               ├── Spur
    │               ├── Mouse Bite
    │               └── Missing Hole
    │
    └───────┬───────┘
            ▼
      Defect Results
```

## Current Progress

### Completed

* [x] PCB image preprocessing
* [x] Image registration
* [x] YOLO model training
* [x] YOLO-based detection of:

  * [x] Open Circuit
  * [x] Short Circuit
  * [x] Spur
  * [x] Mouse Bite
  * [x] Missing Hole
* [x] Initial AOI rule-based inspection development

### In Progress

* [ ] Open Circuit rule-based inspection
* [ ] Connectivity mask generation
* [ ] Connectivity analysis
* [ ] Rule-based Short Circuit detection
* [ ] Rule-based Spur detection
* [ ] Rule-based Mouse Bite detection
* [ ] Rule-based Missing Hole verification
* [ ] Integration of YOLO and rule-based inspection results

## Approach

### 1. Deep Learning

A YOLO object detection model is used as the first layer of inspection.

The model provides:

* Defect classification
* Defect localization
* Bounding boxes
* Confidence scores

This allows the system to quickly identify potential defect regions.

### 2. Rule-Based AOI

Traditional image-processing techniques are used to provide deterministic inspection logic.

Potential techniques include:

* Thresholding
* Morphological operations
* Connected-component analysis
* Contour analysis
* Distance and geometric measurements
* Region masking
* Connectivity analysis

The rule-based layer is intended to complement the YOLO detector rather than completely replace it.

### 3. Connectivity Analysis

Connectivity analysis is being developed to inspect the electrical continuity of PCB traces.

The basic concept is to generate a **connectivity mask** representing conductive regions and analyze whether connected regions follow the expected PCB structure.

This approach is particularly important for detecting defects such as:

* Open circuits
* Short circuits
* Broken traces
* Unwanted connections

## Technologies

* Python
* OpenCV
* NumPy
* YOLO
* PyTorch
* Image Processing
* Computer Vision
* Rule-Based AOI

## Dataset

The initial dataset used for YOLO training was obtained from a publicly available PCB defect dataset on Kaggle.

The model and inspection algorithms are intended primarily for experimentation, learning, and development of PCB AOI techniques.

## Project Status

**Status: In Development**

The deep-learning detection pipeline is functional, while the rule-based AOI layer and connectivity analysis are still under active development.

Future work will focus on improving deterministic defect verification and integrating the different inspection methods into a unified AOI pipeline.

## Future Improvements

* Improve rule-based inspection accuracy
* Complete connectivity-based open circuit detection
* Develop robust short circuit detection
* Improve geometric analysis for spur and mouse bite defects
* Add reference-based defect comparison
* Combine YOLO predictions with rule-based verification
* Add quantitative inspection metrics
* Optimize processing speed for industrial AOI applications

## Disclaimer

This project is a research and development implementation intended to explore PCB Automated Optical Inspection techniques. It is not currently intended to replace production-grade industrial AOI systems.
