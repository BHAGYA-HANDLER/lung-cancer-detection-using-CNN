# Lung Cancer Detection using CNN

A simple deep learning project that detects lung cancer from CT scan images using a Convolutional Neural Network (CNN).

## Overview

This project trains a CNN model to classify lung CT images into three categories:

* Benign (non-cancerous tumor)
* Malignant (cancerous tumor)
* Normal (healthy lung)

The model is trained on the IQ-OTHNCCD Lung Cancer dataset.

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Streamlit (for simple frontend)

## Project Structure

```
lung-cancer-detection/
│
├── IQ-OTHNCCD/          # Dataset
├── app.py               # Streamlit frontend
├── train_model.py       # Model training script
├── lung_cancer_model.h5 # Trained model
├── requirements.tx
```
