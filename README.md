# 🚀 Voice Gender Recognition using a Multi-Layer Perceptron

This project demonstrates the development of a deep learning model to classify a speaker's gender (male or female) based on their voice's acoustic features. The core of the project is a Multi-Layer Perceptron (MLP), a type of feedforward neural network particularly effective for classification tasks on structured, tabular data.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg)

---

## 📖 Project Overview

The goal is to build a robust classifier using a Keras-based neural network. The project covers the entire machine learning workflow: data loading and preprocessing, model architecture design, efficient training with callbacks, and a detailed performance evaluation, including a confusion matrix analysis.

---

## 📊 The Dataset

This project utilizes the **"Gender Recognition by Voice"** dataset from Kaggle.

* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/primarypuzzles/voice-gender)
* **Total Samples:** 3,168 voice recordings.
* **Distribution:** The dataset is perfectly balanced, with 1,584 samples from male speakers and 1,584 from female speakers.

### Acoustic Features

For each voice sample, the following 20 acoustic properties were measured:

* **meanfreq:** mean frequency (in kHz)
* **sd:** standard deviation of frequency
* **median:** median frequency (in kHz)
* **Q25:** first quantile (in kHz)
* **Q75:** third quantile (in kHz)
* **IQR:** interquantile range (in kHz)
* **skew:** skewness
* **kurt:** kurtosis
* **sp.ent:** spectral entropy
* **sfm:** spectral flatness
* **mode:** mode frequency
* **centroid:** frequency centroid
* **peakf:** peak frequency (frequency with highest energy)
* **meanfun:** average of fundamental frequency
* **minfun:** minimum fundamental frequency
* **maxfun:** maximum fundamental frequency
* **meandom:** average of dominant frequency
* **mindom:** minimum of dominant frequency
* **maxdom:** maximum of dominant frequency
* **dfrange:** range of dominant frequency
* **modindx:** modulation index

The final column, **`label`**, indicates the gender of the speaker (`male` or `female`).

---

## 🧠 Model Architecture: Multi-Layer Perceptron (MLP)

This project utilizes a **Multi-Layer Perceptron (MLP)**, a classic yet powerful type of feedforward neural network. An MLP is an excellent choice for this task because it is designed to find complex, non-linear patterns in tabular data, which is exactly the format of our pre-extracted acoustic features.

The architecture consists of:
* An **Input Layer** that accepts the 20 acoustic features for each voice sample.
* Two hidden **`Dense`** layers with **`ReLU`** activation functions, which learn progressively more complex representations of the data.
* **`Dropout`** layers placed after each hidden layer to prevent overfitting by randomly deactivating neurons during training, forcing the network to learn more robust features.
* A final **`Dense`** output layer with a single neuron and a **`sigmoid`** activation function, which outputs a probability score between 0 and 1 for the binary classification task.

---

## ✨ Project Workflow & Features

* **Data Preprocessing:**
    * The labels (`male`/`female`) are encoded into numerical format (1/0).
    * The dataset is split into training (80%) and testing (20%) sets.
    * **Min-Max Scaling** is applied to normalize the features, ensuring the scaler is fit **only** on the training data to prevent data leakage.
* **Training & Optimization:**
    * The model is compiled with the `Adam` optimizer and `binary_crossentropy` loss function.
    * Performance is monitored using `accuracy` and `recall` metrics.
    * Callbacks like **`EarlyStopping`** and **`ReduceLROnPlateau`** are used to optimize training, find the best model weights, and prevent overfitting.
* **Evaluation:**
    * The trained model's performance is measured on the unseen test set.
    * A **Confusion Matrix** is generated to provide a detailed visual breakdown of classification performance.

---

## 📈 Results

The model achieved excellent performance on the test set, demonstrating a strong ability to generalize.

| Metric | Test Set Score |
| :--- | :--- |
| **Accuracy** | 97.79% |
| **Recall** | 97.33% |
| **Loss** | 0.0618 |

### Confusion Matrix Analysis

The confusion matrix provides a clear and detailed breakdown of the model's predictions on the 634 test samples. The results confirm the model's high performance and reliability.

![Confusion Matrix]
* **True Positives (Male): 328** - The model correctly identified 328 male voices.
* **True Negatives (Female): 292** - The model correctly identified 292 female voices.
* **False Positives (Type I Error): 5** - Only 5 female voices were incorrectly classified as male.
* **False Negatives (Type II Error): 9** - Only 9 male voices were incorrectly classified as female.

The high values on the main diagonal (292 and 328) and the very low values on the off-diagonal (5 and 9) demonstrate the model's high precision and recall for both classes.
