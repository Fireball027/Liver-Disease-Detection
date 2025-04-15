## Overview
The **Liver Disease Prediction** project applies **Machine Learning (ML)** to detect the presence of liver disease in patients based on clinical attributes. By using different classification algorithms, the project identifies the most accurate and reliable model to assist in early disease diagnosis.

The dataset used for this project is sourced from Kaggle:
[Indian Liver Patient Dataset](https://www.kaggle.com/uciml/indian-liver-patient-records)

---

## Key Features
- **Data Cleaning and Preprocessing**: Handles missing values and encodes categorical variables.
- **Exploratory Data Analysis (EDA)**: Visualizes feature distributions and correlations.
- **Model Training**: Trains three classification models — SVM, Logistic Regression, and Random Forest.
- **Model Evaluation**: Evaluates models using Accuracy, Precision, Recall, and F1 Score.
- **Visualization**: Displays model performance and feature impact through plots.

---

## Project Files

### 1. `indian_liver_patient.csv`
- Contains 583 observations and 10 clinical features:
  - `Age`, `Gender`, `Total_Bilirubin`, `Direct_Bilirubin`, `Alkaline_Phosphotase`,
  - `Alamine_Aminotransferase`, `Aspartate_Aminotransferase`, `Total_Proteins`,
  - `Albumin`, `Albumin_and_Globulin_Ratio`, `Dataset` (1 = Disease, 2 = No Disease)

### 2. `main.py`
Handles the entire machine learning pipeline from loading the data to evaluating classification models.

#### Key Components:

**1. Data Preprocessing**
- Replaces missing values with the column mean or drops them.
- Converts `Gender` to numerical format.
- Encodes the target class (`Dataset`) as binary (1 = Disease, 0 = No Disease).

**2. Exploratory Data Analysis**
- Plots feature distributions and class balance.
- Uses heatmaps to display feature correlations.

**3. Feature Engineering and Model Training**
- Splits data into training and testing sets.
- Applies **TF-IDF** and scaling as needed.
- Trains the following models:
  - **Support Vector Machine (SVM)**
  - **Logistic Regression**
  - **Random Forest Classifier**

**4. Model Evaluation**
- Compares all models using metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Visualizes confusion matrices and classification reports.

#### Example Code Snippet:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()
```

---

## How to Run the Project

### Step 1: Install Dependencies
```bash
pip install pandas matplotlib seaborn scikit-learn
```

### Step 2: Run the Script
```bash
python main.py
```

### Step 3: View Results
- Compare classifier performance.
- Explore plots showing predictions, confusion matrices, and feature importance.

---

## Future Enhancements
- **Hyperparameter Tuning**: Use Grid Search for optimal model parameters.
- **Model Explainability**: Implement SHAP or LIME for feature impact analysis.
- **Web App**: Deploy as a simple diagnostic tool using Streamlit.
- **Additional Models**: Include Gradient Boosting and Neural Networks.

---

## Conclusion
This project demonstrates how machine learning models can assist in early liver disease detection, providing accurate predictions using clinical data. The comparative model analysis highlights strengths and weaknesses, paving the way for real-world deployment in healthcare.

---

**Stay Healthy, Keep Learning! 🚀**

