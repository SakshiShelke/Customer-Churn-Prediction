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

Built using **Python**, **Streamlit**, and a lot of ☕.

---
---

## 📊 Screenshots(Plots)
![image](https://github.com/user-attachments/assets/55b539ea-1ca8-434c-b836-1657dae83737)
![image](https://github.com/user-attachments/assets/0d52f689-55dc-47c6-ad1a-5a19e704d6a8)
![image](https://github.com/user-attachments/assets/53d91bfc-d580-466e-ade4-de01b35da39d)
![image](https://github.com/user-attachments/assets/4cc975d9-144a-4721-a395-b9929d01e924)
![image](https://github.com/user-attachments/assets/fec3faeb-08f0-44fd-ada3-9900890a90db)
![image](https://github.com/user-attachments/assets/05472148-49ad-40e1-8180-4bfb31101a2d)
![image](https://github.com/user-attachments/assets/a071e752-b1f5-45ee-b4c2-860b96adec5c)
![image](https://github.com/user-attachments/assets/a1ca367e-6cc1-40c1-9ed8-3e6873e1847a)
![image](https://github.com/user-attachments/assets/ff2e3960-2de2-440e-b704-9639ab214fef)
![image](https://github.com/user-attachments/assets/1fd27375-3491-424a-b602-032965123570)
![image](https://github.com/user-attachments/assets/ce80d3d7-30b8-4de9-b82d-3a19a8eb0ee8)
![image](https://github.com/user-attachments/assets/1e4785f9-52ab-40d2-aa5b-c461c45be7aa)

---

## 🖼️ Screenshots(Real Time Prediction)
![image](https://github.com/user-attachments/assets/6b5c31b6-b3f0-425c-bbe9-95ec299c770b)
![image](https://github.com/user-attachments/assets/002ed87a-f8c1-4f5c-9e22-4071410a9f51)
![image](https://github.com/user-attachments/assets/62d936d6-f882-4e66-a2a0-d3f5dd46a8f4)

---
