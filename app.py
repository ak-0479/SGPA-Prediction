import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load and train the model
df = pd.read_csv("SGPA Prediction - Sheet1.csv")
X = df[['Hours of study']]
y = df['SGPA']

model = LinearRegression()
model.fit(X, y)

# App interface
st.title("📘 SGPA Predictor")
st.caption("Made by Ankita Kanjilal")

hours = st.number_input("Enter study hours:", min_value=0.0, max_value=24.0, step=0.5)

if st.button("Predict SGPA"):
    prediction = model.predict([[hours]])[0]
    prediction = min(prediction, 10)  # Ensure SGPA doesn't exceed 10
    st.success(f"🎓 Predicted SGPA: {prediction:.2f}")