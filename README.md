# Plant Leaf Disease Classification

A deployable computer vision application for classifying plant leaf images into supported healthy and diseased categories using handcrafted image features and machine learning models.

**Live MVP:** https://plant-leaf-disease-classification-madkywpuhtywas8cl2kme9.streamlit.app/

---

## Overview

Plant diseases can reduce crop quality and yield when symptoms are not identified early. This project builds an image-based classification pipeline that analyzes leaf images, extracts visual disease-related features, and predicts the most likely class among selected plant leaf categories.

The system combines traditional computer vision techniques with machine learning. It does not rely on deep learning; instead, it uses preprocessing, disease spot segmentation, color features, texture features, and classical classifiers.

---

## Supported Classes

The current model is trained only on selected classes from the New Plant Diseases Dataset:

| Plant | Condition |
|---|---|
| Peach | Bacterial spot |
| Peach | Healthy |
| Pepper bell | Bacterial spot |
| Pepper bell | Healthy |
| Strawberry | Leaf scorch |
| Strawberry | Healthy |

Images outside these plant types may still produce a prediction, but the result should be interpreted carefully because the model was not trained on those categories.

---

## Live Application

The Streamlit MVP allows users to upload a leaf image and receive:

- Predicted plant disease class
- Model confidence score
- Disease area ratio
- Approximate disease spot mask
- Disease spot overlay visualization
- Summary of extracted handcrafted features

Open the deployed app here:

https://plant-leaf-disease-classification-madkywpuhtywas8cl2kme9.streamlit.app/

---

## Methodology

The project follows a handcrafted computer vision pipeline:

1. **Image Loading**  
   Load selected plant leaf images from the dataset.

2. **Preprocessing**  
   Resize images, reduce noise using Gaussian and median filtering, and convert images into HSV and LAB color spaces.

3. **Disease Spot Segmentation**  
   Apply color-based segmentation using HSV/LAB information, Otsu thresholding, K-Means clustering, and morphological operations.

4. **Feature Extraction**  
   Extract handcrafted features, including:
   - Disease area ratio
   - Spot count
   - Largest spot ratio
   - HSV and LAB color moments
   - GLCM texture features
   - LBP texture histogram

5. **Classification**  
   Train classical machine learning models using extracted features.

6. **Evaluation**  
   Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.

7. **Deployment**  
   Package the final inference pipeline into a Streamlit web application.

---

## Models

The repository includes trained model files used by the application:

| Model | File |
|---|---|
| Random Forest | `models/full_random_forest.pickle` |
| XGBoost | `models/full_xgboost.pickle` |

The MVP currently uses the Random Forest model by default.

---

## Repository Structure

```text
plant-leaf-disease-classification/
├── app.py
├── Plant_Disease_Classification.ipynb
├── README.md
├── requirements.txt
├── train_handcrafted_features.csv
├── valid_handcrafted_features.csv
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   ├── full_random_forest.pickle
│   └── full_xgboost.pickle
│
├── plant_classifier/
│   ├── __init__.py
│   ├── inference.py
│   └── model_store.py
│
└── outputs/
    └── figures/
```

Dataset folders are intentionally excluded from GitHub because the raw image dataset is large.

---

## Dataset

This project uses selected classes from the **New Plant Diseases Dataset** available on Kaggle:

https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Only six classes are used in the current version:

```text
Peach___Bacterial_spot
Peach___healthy
Pepper,_bell___Bacterial_spot
Pepper,_bell___healthy
Strawberry___Leaf_scorch
Strawberry___healthy
```

The raw dataset is not included in this repository. To reproduce training locally, download the dataset from Kaggle and place it in the expected local dataset folder structure used by the notebook.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/LecyLecy/plant-leaf-disease-classification.git
cd plant-leaf-disease-classification
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

For Windows:

```bash
.venv\Scripts\activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application Locally

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Running the Notebook

Open the notebook:

```bash
jupyter notebook Plant_Disease_Classification.ipynb
```

The notebook contains the full pipeline, including data loading, preprocessing, feature extraction, model training, evaluation, and model saving.

---

## Tech Stack

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Pillow

---

## Limitations

This project is an MVP and should not be treated as a production-grade agricultural diagnosis system.

Key limitations:

- The model only supports Peach, Pepper Bell, and Strawberry leaf categories.
- Images from unsupported plants may produce unreliable predictions.
- Segmentation is approximate and may be affected by lighting, background, shadows, and image quality.
- The model uses handcrafted features, so it may not generalize as well as a properly trained deep learning model on a larger and more diverse dataset.
- Confidence scores should be interpreted as model uncertainty indicators, not absolute correctness guarantees.

---

## Future Improvements

Potential improvements include:

- Add more plant species and disease categories
- Improve leaf/background segmentation
- Add out-of-distribution detection for unsupported plant types
- Train a CNN or transfer learning model for stronger image recognition performance
- Add confidence thresholding and user guidance for uncertain predictions
- Deploy a lighter model for faster cloud inference

---

## Disclaimer

This application is intended for educational, experimental, and demonstration purposes. It should not replace expert agricultural inspection or professional plant disease diagnosis.
