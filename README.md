# Breast Cancer Detection CNN Portal

### Developed by: Eyerusalem Gebremdhin
**Modality:** Screening Mammography (X-ray)  
**Architecture:** Pre-trained ResNet50 Backbone (Transfer Learning) via TensorFlow/Keras  
**Interface:** Interactive Streamlit Web Portal  

---

## Project Overview
This repository contains a complete computer vision framework built to assist in the binary classification of screening mammograms into **Benign** or **Malignant** categories. The core pipeline utilizes a pre-trained ResNet50 Convolutional Neural Network (CNN) with frozen backbone layers to capture complex visual textures and anatomical tissue density shifts without running into data leakage or overfitting traps.

---

## Repository Structure
* `Assignment-computer vision.ipynb` — Complete pipeline covering exploratory data analysis, class distribution visualizations, patient-grouped splitting data validation, model training history curves, and sample scan visualizations.
* `app.py` — The user interface script powering the interactive web application.
* `breast_mammography_model.h5` — Compressed compilation of final trained network layers.
* `requirements.txt` — Explicit configuration manifest mapping all required dependency versions.

---

## Local Deployment Guide
To initialize this system on a local machine, execute the following steps in sequence:

1. Clone the repository and navigate into the workspace:
   ```bash
   git clone https://github.com
   cd breast-cancer-detection-cnn
   ```

2. Synchronize all software components using the version list file:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the Streamlit diagnostics engine web portal:
   ```bash
   streamlit run app.py
   ```

---

## Academic Disclaimer
This application is a student coursework prototype built exclusively for an AkiraChix academic assignment. It is not an FDA-approved medical diagnostic tool and must not be used for actual clinical decisions.
