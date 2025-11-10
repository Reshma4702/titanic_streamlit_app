import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details below to predict survival:")

# Create input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Prediction button
if st.button("Predict"):
    sex = 0 if sex == "male" else 1
    embarked = {"S": 0, "C": 1, "Q": 2}[embarked]
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ The passenger **survived!**")
    else:
        st.error("❌ The passenger **did not survive.**")
