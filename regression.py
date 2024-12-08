import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('notebooks/regression_model.h5') 

with open('notebooks/lle_gender.pkl','rb') as file:
    lle_gender = pickle.load(file)

with open('notebooks/ohe_geog.pkl','rb') as file:
    ohe_geog = pickle.load(file)

with open('notebooks/scaler.pkl','rb') as file:
    scaler = pickle.load(file)  
 


## streamlit app
st.title('Estimated Salary')

# User input
geography = st.selectbox('Geography', ohe_geog.categories_[0])
gender = st.selectbox('Gender', lle_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0,1]) 
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lle_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited' : [exited]  
})

# One-hot encode 'Geography'
geo_encoded = ohe_geog.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geog.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict estimated salary 
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.write(f'Churn Probability: ${predicted_salary:.2f}')


 