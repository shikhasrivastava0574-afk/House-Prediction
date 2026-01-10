import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction Dashboard")

# ------------------ FILE CHECK ------------------
if not os.path.exists("house_price_model.pkl"):
    st.error("❌ house_price_model.pkl not found")
    st.stop()

if not os.path.exists("model_columns.pkl"):
    st.error("❌ model_columns.pkl not found")
    st.stop()

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("house_price_model.pkl", "rb"))
columns = pickle.load(open("model_columns.pkl", "rb"))

st.success("Model loaded successfully!")

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("🧾 Enter House Details")

OverallQual = st.sidebar.slider("Overall Quality (1–10)", 1, 10, 5)
OverallCond = st.sidebar.slider("Overall Condition (1–10)", 1, 10, 5)
GrLivArea = st.sidebar.number_input("Living Area (sq ft)", 500, 5000, 1500)
GarageCars = st.sidebar.slider("Garage Cars", 0, 4, 2)
TotalBsmtSF = st.sidebar.number_input("Basement Area (sq ft)", 0, 3000, 800)
YearBuilt = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
FullBath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
BedroomAbvGr = st.sidebar.slider("Bedrooms", 0, 6, 3)

# ------------------ MAIN LAYOUT ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Prediction Panel")

    if st.button("Predict Price"):
        input_data = pd.DataFrame(0, index=[0], columns=columns)

        input_data["OverallQual"] = OverallQual
        input_data["OverallCond"] = OverallCond
        input_data["GrLivArea"] = GrLivArea
        input_data["GarageCars"] = GarageCars
        input_data["TotalBsmtSF"] = TotalBsmtSF
        input_data["YearBuilt"] = YearBuilt
        input_data["FullBath"] = FullBath
        input_data["BedroomAbvGr"] = BedroomAbvGr

        pred = model.predict(input_data)
        pred = np.expm1(pred)[0]

        # Confidence range ±10%
        low = pred * 0.90
        high = pred * 1.10

        st.success(f"💰 Estimated House Price: ₹ {pred:,.2f}")
        st.info(f"📊 Expected Range: ₹ {low:,.2f}  —  ₹ {high:,.2f}")

with col2:
    st.subheader("📈 Model Insights")

    # Feature Importance (Top 10)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_df = pd.DataFrame({
            "Feature": columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        st.write("Top 10 Important Features")
        st.bar_chart(feature_df.set_index("Feature"))

# ------------------ SALE PRICE DISTRIBUTION ------------------
st.subheader("📊 Sale Price Distribution (Training Data)")

if os.path.exists("train.csv"):
    data = pd.read_csv("train.csv")

    fig, ax = plt.subplots()
    ax.hist(data["SalePrice"], bins=40)
    ax.set_title("Distribution of Sale Prices")
    ax.set_xlabel("Price")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.warning("train.csv not found. Upload it to show price distribution.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("🚀 Built using Streamlit + Random Forest Regressor")
st.markdown("By Shikha Srivastava")
