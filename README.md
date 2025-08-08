# Breast Cancer Diagnosis - Logistic Regression Classifier

This project uses **Logistic Regression** to perform **binary classification** on the **Breast Cancer Wisconsin Dataset**. The goal is to predict whether a tumor is **malignant (0)** or **benign (1)** based on various features derived from digitized images of breast masses.

---

## 📁 Dataset Info

- Source: `sklearn.datasets.load_breast_cancer()`
- Classes: `Malignant = 0`, `Benign = 1`
- Features: 30 numeric features like radius, texture, smoothness, symmetry, etc.

---

## 🔧 Tools & Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (sklearn)

---

##  Project Workflow

### 1. Load Dataset
- Loaded breast cancer dataset using `sklearn.datasets`

### 2. Preprocess
- Split dataset into training and testing sets (80/20)
- Applied `StandardScaler()` to normalize feature values

### 3. Train Model
- Used `LogisticRegression()` to train a binary classifier

### 4. Evaluate Model
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- ROC Curve & AUC Score
- Threshold Tuning (custom threshold = 0.6)

### 5. Output
- Final predictions exported as `diagnostic_results.csv` with:
  - Scaled test features
  - Actual target values
  - Predicted labels
  - Predicted probabilities (sigmoid output)

---

## 📉 Visualizations

- Confusion Matrix (default threshold 0.5)
- ROC Curve with AUC score
- Sigmoid function plot
- Confusion Matrix (after threshold tuning to 0.6)

---

## 📁 Output File

- `diagnostic_results.csv` contains:
  - Standardized test data
  - Actual labels
  - Predicted labels
  - Probability of class 1 (benign)

---

## 📸 Screenshots (Optional)
- ROC curve
- Confusion matrix
- Sigmoid function plot
- `diagnostic_results.csv` preview

---

## 🧠 Author

- **Name**: P.Narasimha Murthy
- **Internship Program**: AI & ML Internship Task 4

---

## 📌 Task Instructions Reference

- Task 4: Classification with Logistic Regression  
- From: Internship Guidelines  
- Tool: Scikit-learn, Matplotlib, Pandas  
- Dataset: Breast Cancer Wisconsin Dataset

---

## 📤 Submission

> 🔗 GitHub Repo Link: [paste your repository link here]  
> 📄 Submit using the form link provided in your internship instructions.
