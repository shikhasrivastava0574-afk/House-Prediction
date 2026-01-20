# 🏠 House Price Prediction Dashboard

An end-to-end Machine Learning web application that predicts house prices using a trained Random Forest model and provides real-time predictions through a Streamlit dashboard.

This project covers the complete ML lifecycle:  
**Data → Model → Web App → Deployment**

---

## 📌 Project Overview

This application predicts the market price of a house based on important property features such as:

- Overall quality  
- Living area  
- Basement area  
- Garage capacity  
- Year built  
- Number of bedrooms & bathrooms  

The model is trained on Kaggle’s  
**House Prices – Advanced Regression Techniques** dataset and deployed using **Streamlit Community Cloud**.

---

## 🧠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- Pickle  
- GitHub  
- Streamlit Cloud  

---

## 🏗️ Project Architecture
Dataset → Preprocessing → Model Training → Model Saving (.pkl)
→ Streamlit UI → GitHub → Streamlit Cloud Deployment


---

## 📂 Project Structure
house-price-prediction-app/
│
├── app.py
├── house_price_model.pkl
├── model_columns.pkl
├── train.csv
├── requirements.txt
├── README.md
└── images/
├── dashboard_home.png
├── prediction_panel.png
├── feature_importance.png
└── price_distribution.png


---

## 🖥️ Dashboard Features

- 🎛 Sidebar-based input form  
- ⚡ Real-time price prediction  
- 📊 Prediction confidence range (±10%)  
- 📈 Feature importance visualization  
- 📉 Sale price distribution plot  
- 🚀 Fully deployed web application  

---

## 📸 Screenshots

> Add your screenshots inside the `images/` folder and use the names below.

### 🏠 Dashboard Home
```md
![Dashboard Home](images/dashboard_home.png)

```
### 🎯 Prediction Panel
```md
![Prediction Panel](images/prediction_panel.png)
```

### 📊 Feature Importance
```md
![Feature Importance](images/feature_importance.png)
```

### 📉 Sale Price Distribution
```md
![Sale Price Distribution](images/price_distribution.png)
```

### 🔍 How It Works

User enters house details in the sidebar

Input is transformed into model-compatible format

Random Forest model predicts log-transformed price

Output is converted back to real price

Final price and confidence range is displayed

### 📈 Model Details

Algorithm: Random Forest Regressor

Target Variable: SalePrice

Target Transformation: Logarithmic

Evaluation Metric: RMSE

### 🌐 Live Application
🔗 https://house-prediction-dmfp9qvtymxckhjfmoade8.streamlit.app/


(Replace with your deployed Streamlit URL)

### 🧾 Run Locally
pip install -r requirements.txt
streamlit run app.py
