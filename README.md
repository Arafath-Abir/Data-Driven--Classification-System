# Data Preprocessing & PCA for Income Classification

This project demonstrates a complete **data preprocessing and classification pipeline** in Python.  
It focuses on preparing raw data, applying dimensionality reduction (PCA), and training a machine learning model to predict income categories.

---

## 🚀 Features

- 🧹 **Data Preprocessing**
  - Handling missing values
  - Encoding categorical variables
  - Scaling and normalization
  - Train/test split

- 📉 **Dimensionality Reduction**
  - Applying **Principal Component Analysis (PCA)**
  - Visualizing reduced features

- 🤖 **Machine Learning**
  - Logistic Regression classifier
  - Model training and evaluation
  - Metrics: Accuracy, Confusion Matrix, ROC Curve

---

## 📂 Project Structure

```
.
├── notebooks/
│   └── Data_Preprocess.ipynb   # Jupyter Notebook with code & explanations
├── src/
│   └── preprocess_pipeline.py  # Optional clean script version
├── data/
│   └── data.csv                # Example dataset (replace with your own)
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Technologies Used

- **Python 3**
- Libraries:
  - `pandas` – data handling
  - `numpy` – numerical computations
  - `scikit-learn` – preprocessing, PCA, classification
  - `matplotlib`, `seaborn` – visualization

---

---

## 📸 Example Workflow

**1. Load and preprocess the dataset**
```python
import pandas as pd

df = pd.read_csv("data/data.csv")
print(df.isnull().sum())
```

**2. Apply PCA**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop("income", axis=1))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

**3. Train a classifier**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_pca, df["income"], test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
```

---

## 🎯 Learning Outcomes

- Gained experience in **data preprocessing** workflows  
- Applied **dimensionality reduction (PCA)** on real data  
- Built and evaluated a **classification model**  
- Improved understanding of end-to-end **ML pipelines**  
