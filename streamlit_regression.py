import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import pickle

#Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

#Load the encoder and scalers
with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Steamlit  app
st.title("Estimated Salary Prediction")

#User input
credit_score=st.number_input('Credit Score')
geography=st.selectbox('Geography', label_encoder_geo.categories_[0])
gender=st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider('Age', 0, 18, 91)
tenure=st.slider('Tenure', 0, 10)
balance=st.number_input('Balance')
num_of_products=st.slider('Number of Products', 1, 4)
has_cr_card=st.selectbox('Has Credit Card', [0, 1])
is_active_member=st.selectbox('Is Active Member', [0, 1])
exited=st.selectbox('Exited', [0, 1])

#Store the input data in a DataFrame
input_data= pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

#One-hot encode for geography
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

#Combine the encoded features with the rest of the input data
input_data= pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scaling the input data
input_data_scaled= scaler.transform(input_data)

#Predict estimated salary
prediction= model.predict(input_data_scaled)
prediction_salary= prediction[0][0]

st.write(f"Predicted Estimated Salary: {prediction_salary:.2f}")

