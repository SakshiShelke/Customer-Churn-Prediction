---

# 📉 Customer Churn Prediction Dashboard

An interactive, end-to-end machine learning dashboard built with **Streamlit** to predict telecom customer churn using a **Gradient Boosting Classifier**. This tool allows you to

✅ Upload customer data
🧠 Train a model
🔮 Predict churn
📊 Visualize key insights
💡 Explain predictions using SHAP values

---

## 🚀 Features

* ✅ Upload and preprocess Telco customer CSV files
* 🧠 Train a **Gradient Boosting Classifier**
* 🔮 Make predictions on new/unseen customer data
* 📈 Visualize:

  * Churn distribution (pie/bar)
  * SHAP feature importance
  * Cost analysis
  * ROC curves (overall and by tenure)
  * Churn by contract type
  * Violin & scatter plots
* 💡 SHAP explanations for both batch and real-time predictions
* 🎯 Real-time churn prediction input form in the sidebar
* 📥 Download predictions as CSV

---

## 📂 File Structure

```
stop-the-churn/
│
├── dashboard.py         # Main Streamlit app
├── requirements.txt     # Project dependencies
├── telco_train.csv      # Sample training dataset
├── telco_test.csv       # Sample test dataset
└── README.md            # This file
```

---

## 🧾 Dataset Requirements

### Training Data:

* Format: CSV
* Required columns:

  * `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`
  * Target column: `Churn` (values: Yes/No)
* Optional: Features like `gender`, `Partner`, `Contract`, `OnlineSecurity`, etc.

### Test Data:

* Same feature columns as training data (without `Churn`)

---

## 🛠 Getting Started

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run dashboard.py
```

---

## 🧠 Machine Learning Model

* **Algorithm:** `GradientBoostingClassifier` (from `sklearn.ensemble`)
* **Target:** Binary churn classification (`Churn` mapped to `churned`)
* **Categorical Encoding:** LabelEncoder
* **Evaluation Metric:** ROC-AUC Score

---

## 📊 Visualizations

* Churn vs. Retention (Pie chart)
* Churn Probability Histogram
* ROC Curve (overall + tenure-based)
* SHAP Summary Plot
* Cost Distribution (Box plot)
* Churn by Contract Type (Bar + Violin)
* Tenure vs Churn Probability (Scatter)

---

## 🧮 Real-Time Churn Prediction

* Accessible via the **sidebar**
* Fill out customer feature fields
* Instantly receive churn probability and prediction
* Visualize SHAP waterfall plot for the specific customer

---

## 📥 Input/Output Format

### Input

* **Training CSV** (must contain `Churn` column)
* **Test CSV** (no `Churn` column required)

### Output

* Prediction CSV with:

  * `customerID`, `Probability`, `Churn_Predicted`, `Retain`, `cost`, `contract`

---

## 📌 Example Workflow

1. Upload your **training data** with a `Churn` column
2. Click to **train the model**
3. Review training metrics and SHAP importance
4. Upload **test data** to generate predictions
5. Download predictions and explore visual insights
6. Use the **sidebar** to predict churn for a single customer in real-time

---

## 🧰 Dependencies

```
streamlit
pandas
numpy
seaborn
matplotlib
plotly
shap
scikit-learn
ipython
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 🌱 Future Enhancements

* Add support for model comparison (Random Forest, XGBoost)
* Auto hyperparameter tuning
* Customer retention strategy generator
* Save trained model for reuse

---

## 👩‍💻 Authors

* **Sakshi Shelke**
* **Khush Aghera**
* **Yash Kumar**
* **Darshan Chhatbar**
* **Om Pandey**

Built using **Python**, **Streamlit**, and a lot of ☕.

---
