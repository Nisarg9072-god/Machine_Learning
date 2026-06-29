<div align="center">

# 🤖 Machine Learning Path

### A structured, hands-on journey through core ML concepts — from raw data to deployable models.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 About

This repository documents my end-to-end learning path in Machine Learning — every folder is a standalone unit covering a specific concept, algorithm, or real-world project. Code is deliberately kept readable so it also serves as a reference.

---

## 🗂️ Repository Structure

```
Machine_Learning/
│
├── Data_Cleaning_And_Processing/       ← Foundation: how to prepare raw data
│   ├── data_processing.py             ← Core preprocessing pipeline demo
│   ├── insurance.ipynb                ← Insurance EDA (Jupyter Notebook)
│   ├── Heart.ipynb                    ← Heart Disease EDA (Jupyter Notebook)
│   ├── Project1/                      ← Insurance dataset: cleaning + encoding
│   │   ├── main.py
│   │   └── insurance.csv
│   └── Project_2/                     ← Heart disease dataset: full EDA + prep
│       ├── main.py
│       ├── heart.csv
│       └── Figure_1~5.png             ← Generated EDA plots
│
└── ML1/                               ← Supervised & Unsupervised algorithms
    ├── Linear_regression_mode/        ← Linear Regression projects
    │   ├── Insurance/                 ← Predicting insurance charges
    │   │   ├── main.py
    │   │   └── insurance.csv
    │   └── Ford_car_price model/      ← Predicting Ford car prices
    │       ├── main.py
    │       └── ford.csv
    ├── Iris_project/                  ← Classification on Iris dataset
    │   └── iris-dataset.py
    └── KMeans_clustering_project/     ← Customer segmentation with K-Means
        └── KMeans_Clustering.py
```

---

## 🧩 Modules at a Glance

### 📦 Module 1 — Data Cleaning & Processing

> *"Garbage in, garbage out."*  Clean data is the most important step.

| Topic | File | Key Skills |
|---|---|---|
| Preprocessing Pipeline | `data_processing.py` | Missing values, Label Encoding, Standard Scaler, train/test split |
| Insurance EDA (Script) | `Project1/main.py` | Duplicates, one-hot encoding, feature engineering (BMI category) |
| Insurance EDA (Notebook) | `insurance.ipynb` | Interactive EDA with Jupyter |
| Heart Disease EDA (Script) | `Project_2/main.py` | Cholesterol imputation, get_dummies, multi-column scaling |
| Heart Disease EDA (Notebook) | `Heart.ipynb` | Full visual EDA with generated plots |

---

### 🤖 Module 2 — ML Algorithms (ML1/)

#### 📈 Linear Regression — Insurance Charges Prediction
- **Dataset:** [Kaggle Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Goal:** Predict medical insurance charges from age, BMI, smoker status etc.
- **Techniques:** Feature encoding, Standard Scaler, `LinearRegression`
- **Folder:** `ML1/Linear_regression_mode/Insurance/`

#### 🚗 Linear Regression — Ford Car Price Prediction
- **Dataset:** Ford Car Price dataset (`ford.csv`)
- **Goal:** Predict the price of a Ford car from its features (year, mileage, engine size etc.)
- **Techniques:** Feature encoding, `LinearRegression`, model evaluation
- **Folder:** `ML1/Linear_regression_mode/Ford_car_price model/`

#### 🌸 Classification — Iris Flower Species
- **Dataset:** Scikit-learn built-in Iris dataset (150 samples, 3 classes)
- **Goal:** Classify flowers into *Setosa / Versicolor / Virginica*
- **Techniques:** `DecisionTreeClassifier`, train/test split, `accuracy_score`
- **Folder:** `ML1/Iris_project/`

#### 🔵 Unsupervised — K-Means Customer Segmentation
- **Dataset:** Mall Customers CSV
- **Goal:** Segment customers by Annual Income & Spending Score
- **Techniques:** `StandardScaler`, Elbow Method, `KMeans`, Silhouette Score, cluster visualisation
- **Folder:** `ML1/KMeans_clustering_project/`

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Nisarg9072-god/Machine_Learning.git
cd Machine_Learning
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run any script
```bash
# Example — Insurance Linear Regression
python ML1/Linear_regression_mode/Insurance/main.py

# Example — Ford Car Price Prediction
python "ML1/Linear_regression_mode/Ford_car_price model/main.py"

# Example — Iris classification
python ML1/Iris_project/iris-dataset.py

# Example — K-Means clustering
python ML1/KMeans_clustering_project/KMeans_Clustering.py
```

---

## 📊 Results Summary

| Project | Algorithm | Metric | Score |
|---|---|---|---|
| Iris Classification | Decision Tree | Accuracy | **~97 %** |
| Customer Segmentation | K-Means (k=5) | Silhouette Score | **~0.55** |
| Insurance Prediction | Linear Regression | R² Score | In progress |
| Ford Car Price Prediction | Linear Regression | R² Score | In progress |

---

## 🛣️ Learning Roadmap

- [x] **Data Preprocessing** — Handling nulls, encoding, scaling, splitting
- [x] **Supervised Learning** — Linear Regression (Insurance & Ford Car Price), Decision Tree Classification
- [x] **Unsupervised Learning** — K-Means Clustering, Elbow Method
- [ ] **Model Evaluation** — Cross-validation, confusion matrix, ROC-AUC
- [ ] **Ensemble Methods** — Random Forest, Gradient Boosting (XGBoost)
- [ ] **Neural Networks** — Intro with TensorFlow / PyTorch
- [ ] **End-to-End Project** — Data → Model → REST API → Deployment

---

## 🔑 Key Concepts Covered

| Concept | Description |
|---|---|
| `StandardScaler` | Zero mean, unit variance normalisation |
| `LabelEncoder` | Ordinal encoding for categorical features |
| `get_dummies` | One-hot encoding with pandas |
| `train_test_split` | Stratified splitting for unbiased evaluation |
| `LinearRegression` | OLS-based regression for continuous targets |
| `DecisionTreeClassifier` | Tree-based classification, interpretable |
| `KMeans` | Centroid-based unsupervised clustering |
| Elbow Method | Optimal K selection via inertia plot |
| Silhouette Score | Cluster quality metric (−1 to +1) |

---

## 📚 Resources

| Resource | Link |
|---|---|
| Scikit-learn Documentation | https://scikit-learn.org/stable/ |
| Pandas User Guide | https://pandas.pydata.org/docs/ |
| Kaggle Datasets | https://www.kaggle.com/datasets |
| Towards Data Science | https://towardsdatascience.com |
| StatQuest (YouTube) | https://www.youtube.com/@statquest |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ as part of a self-taught Machine Learning journey.</sub>
</div>
