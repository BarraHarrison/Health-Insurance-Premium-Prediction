# 📈 Predicting Health Insurance Premiums

## 🏥 Introduction
This project focuses on predicting health insurance premiums based on various factors such as age, BMI, smoking status, and region. Using machine learning techniques, I preprocessed the dataset, explored feature correlations, and trained a **RandomForestRegressor** model to make accurate predictions. Additionally, I fine-tuned the model using **GridSearchCV** to optimize performance.

---

## 🔄 One-Hot Encoding
Before training the model, categorical variables must be converted into numerical values. **One-Hot Encoding** is used to transform non-numeric data into a format suitable for machine learning algorithms.

### **Implementation:**
```python
# Encode categorical variables
data_frame["sex"] = data_frame["sex"].apply(lambda x: 1 if x == "male" else 0)
data_frame["smoker"] = data_frame["smoker"].apply(lambda x: 1 if x == "yes" else 0)
```
Alternatively, **pandas' `get_dummies()`** function is used to encode multiple categorical features efficiently:
```python
import pandas as pd
data_encoded = pd.get_dummies(data_frame, drop_first=True)
```
This ensures that all categorical data is converted into numerical values, making them ready for model training.

---

## 🌡️ Heatmap Correlations
To understand relationships between different features, I used a **heatmap** to visualize correlations in the dataset.

### **Implementation:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,10))
sns.heatmap(data_encoded.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()
```
- **Red shades indicate positive correlations** (as one variable increases, the other also increases).
- **Blue shades indicate negative correlations** (as one variable increases, the other decreases).
- This helps identify which features have a strong influence on health insurance charges.

---

## 🌲 RandomForestRegressor Model
A **RandomForestRegressor** is an ensemble learning method that consists of multiple decision trees. It works by aggregating predictions from several trees to improve accuracy and reduce overfitting.

### **Implementation:**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Split features and the target variable
x = data_encoded.drop("charges", axis=1)
y = data_encoded["charges"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor Model
model = RandomForestRegressor(n_jobs=-1, random_state=42)
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
```
✅ The **RandomForestRegressor** reduces variance and enhances prediction accuracy compared to a single decision tree.

---

## 🎯 Hyperparameter Tuning with GridSearchCV
**GridSearchCV** is used to find the optimal hyperparameters for the model by systematically testing different parameter combinations and selecting the best-performing one.

### **Implementation:**
```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    "max_depth": [None, 2, 7],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [1, 2, 4, 8]
}

# Perform Grid Search
model = RandomForestRegressor(n_jobs=-1)
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)
```
📌 **How GridSearchCV works?**
- It systematically tries different combinations of hyperparameters.
- Uses **cross-validation (cv=5)** to evaluate performance.
- Selects the best parameters to improve model accuracy and prevent overfitting.

---

## 🏁 Conclusion
This project demonstrates how **machine learning models** can be trained to predict **health insurance premiums** based on key factors. Through **data preprocessing, correlation analysis, model training, and hyperparameter tuning**, I improved model performance and enhance prediction accuracy. Future improvements may include:
- Trying other models like **Gradient Boosting** or **XGBoost**.
- Expanding feature engineering for better insights.
- Experimenting with feature scaling and outlier detection.

🚀 **This project provides a solid foundation for using machine learning in predictive analytics!**

