# Logistic Regression Web App

This project implements a logistic regression model using Python and deploys it as a web app using Streamlit.

## 1. Clone the Repository

git clone https://github.com/vikeshkumar8210/Logistic_regression.git
cd Logistic_regression

## 2. Install Requirements
pip install -r requirements.txt
### 3. Train and Save Model
Open and run Logistic_Regression.ipynb in Jupyter Notebook.

This will train the logistic regression model and generate logistic_model.pkl.

## 4. Run Locally with Streamlit
streamlit run streamlit_app.py

## 5. Deploy on Streamlit Cloud
Push all files to GitHub.

Go to https://share.streamlit.io.

Click "New app".

Connect your GitHub account and select this repository.

Choose the branch (e.g., main) and file path: streamlit_app.py.

Click Deploy.

## Notes
Make sure the trained model file logistic_model.pkl is present in the repository.

Ensure requirements.txt contains the following dependencies:

pandas

scikit-learn

streamlit

joblib

###  Screenshot

![Login Page](https://github.com/vikeshkumar8210/Logistic_regression/blob/main/Streamlit.png)
