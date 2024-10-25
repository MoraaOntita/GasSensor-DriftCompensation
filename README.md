# Gas Sensor Array Drift Dataset for Gas Classification

Welcome to the Gas Sensor Array Drift Dataset for Gas Classification project! This repository leverages machine learning techniques to classify six different gases using data from 16 chemical sensors. The dataset, collected over a span of 36 months, aims to tackle sensor drift and create robust gas discrimination models.

## 🧬 Project Overview
In this project, we focus on drift compensation for gas sensor arrays. Sensor drift, an evolving challenge in sensor-based measurements, results from gradual changes in sensor response over time, impacting data quality. The dataset includes thousands of measurements at various concentration levels for six target gases:

1. Ethanol  
2. Ethylene  
3. Ammonia  
4. Acetaldehyde  
5. Acetone  
6. Toluene  

Our model's objective is to learn these drift patterns and perform accurate classification, even as sensor responses change over time.

## 📂 Dataset Information
The dataset consists of 13,910 instances collected by the ChemoSignals Laboratory at UC San Diego, with each sample containing sensor responses to a particular gas. Each sensor reading is represented by a 128-dimensional feature vector, formed by two main feature types:

- **Steady-state features (ΔR):** Capturing resistance change upon exposure.
- **Dynamic features (EMA):** Representing transient behavior using Exponential Moving Average (EMA) values.

### Dataset Structure
The data is organized across 10 batches to represent time periods and gas types for drift analysis:

| Batch ID | Month IDs |
|----------|-----------|
| 1        | Months 1 and 2 |
| 2        | Months 3, 4, 8, 9, and 10 |
| ...      | ...       |
| 10       | Month 36  |

### Key Statistics
- **Instances:** 13,910  
- **Features:** 128  
- **Sensor Types:** 16  
- **Duration:** 36 months  

## 🚀 Goals and Applications
The aim is to improve the classification performance and resilience of gas discrimination tasks over time, enabling the following:

- **Sensor Drift Mitigation:** Detecting and compensating for sensor drift.
- **Gas Discrimination:** Classifying six gases at various concentrations.
- **Feature Engineering:** Leveraging steady-state and dynamic features for classification.

> **Note:** This dataset is intended for academic and research use only and should not be used for commercial purposes.

## ⚙️ Data Preprocessing
Each measurement results in a 128-dimensional feature vector that incorporates the following:

- **Steady-state values (ΔR):** The change between maximum resistance and baseline.
- **Normalized ΔR:** Expressed as a ratio.
- **EMA features (for rising and decaying):** Three different α values (0.001, 0.01, 0.1) capture transient portions in both the increase and decrease phases of each sensor’s response.

## 🧩 Classification Model
To replicate the results from the cited paper, here are the key parameters:

- **Cross-validation folds:** 10  
- **Log-scaled C values:** Range from -5 to 10 with step 1  
- **Log-scaled Gamma (γ) values:** Range from -10 to 5 with step 1  
- **Feature scaling:** Standardize feature values between -1 and +1  

### Training Hyperparameters

| Batch | C    | Gamma      | Accuracy (%) |
|-------|------|------------|--------------|
| 1     | 256.0| 0.03125    | 98.88        |
| 2     | 64.0 | 0.00390625 | 99.76        |
| ...   | ...  | ...        | ...          |
| 10    | 1024.0 | 0.0078125 | 99.66       |

## 🛠️ Project Structure
The following outlines the directory and file structure:

```plaintext
.
├── data                 # Raw data and processed files
├── src                  # Source code for ingestion, preprocessing and model training
├── artifacts            # Trained models and pipeline outputs
├── app.py               # Flask application for predictions
└── README.md            # Project documentation
```

## 🔧 Setup and Installation

To set up this project, follow these steps:

1. Clone the repository:

```bash 
git clone https://github.com/yourusername/gas_sensor_project.git

```

2. Install dependencies:

```bash 
pip install -r requirements.txt

```

3. Run the Flask App:

```bash
python app.py

```

## 📊 Results and Evaluation

The model demonstrates strong classification performance across all batches, achieving an accuracy of over 99% in most cases. Evaluation is done with a 10-fold cross-validation setup, and the classifier is fine-tuned to adjust for varying sensor conditions and gas concentrations.

## 🌐 Contributing
If you'd like to contribute, please open a pull request or raise an issue to discuss. Contributions are welcome for additional feature engineering, alternative model approaches, or enhanced drift handling strategies.