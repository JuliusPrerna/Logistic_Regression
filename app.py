import joblib
import pandas as pd
import streamlit as st

model = joblib.load('logistic_model.pkl')

st.title('Titanic Survival Prediction')
Pclass = st.selectbox('Passenger Class', [1, 2, 3])
Age = st.slider('Age', 0, 80, 30)
SibSp = st.number_input('Siblings/Spouses Aboard', 0, 8, 0)
Parch = st.number_input('Parents/Children Aboard', 0, 6, 0)
Fare = st.slider('Fare', 0.0, 500.0, 32.2)
Sex_male = st.selectbox('Sex', ['Female', 'Male']) == 'Male'
Embarked_Q = st.selectbox('Embarked Q?', [True, False])
Embarked_S = st.selectbox('Embarked S?', [True, False])

features = pd.DataFrame({
    'Pclass': [Pclass],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Sex_male': [int(Sex_male)],
    'Embarked_Q': [int(Embarked_Q)],
    'Embarked_S': [int(Embarked_S)]
})

if st.button('Predict'):
    result = model.predict(features)
    st.write('Prediction:', 'Survived' if result[0] == 1 else 'Not Survived')