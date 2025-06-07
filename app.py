import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("logistic_model.pkl")

st.title("Titanic Survival Predictor")

# Input fields
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Age = st.number_input("Age")
SibSp = st.number_input("Siblings/Spouses Aboard (SibSp)", step=1)
Parch = st.number_input("Parents/Children Aboard (Parch)", step=1)
Fare = st.number_input("Fare")

Sex_male = st.selectbox("Sex", ["Male", "Female"])
Sex_male = 1 if Sex_male == "Male" else 0

Embarked = st.selectbox("Port of Embarkation", ["S", "Q", "C"])
Embarked_Q = 1 if Embarked == "Q" else 0
Embarked_S = 1 if Embarked == "S" else 0

# Prepare feature array
features = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S]])

# Make prediction
if st.button("Predict Survival"):
    prediction = model.predict(features)
    st.write("Prediction:", "Survived" if prediction[0] == 1 else "Did Not Survive")
