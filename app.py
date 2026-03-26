import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# load the trained model
model = tf.keras.models.load_model('model.h5')

# load the encoders and scaler
with open('onehot_encoder_Geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# streamlit app
st.title('Customer Churn Prediction')

# user input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
tenure = st.slider('Tenure (years)', 0, 10)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

if st.button('Predict'):
    # build input dataframe
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Gender': label_encoder_gender.transform([gender])[0],
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }])

    # one-hot encode Geography
    geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

    input_data = pd.concat([input_data, geo_df], axis=1)

    # scale
    input_scaled = scaler.transform(input_data)

    # predict
    prediction = model.predict(input_scaled)
    probability = prediction[0][0]

    st.write(f'Churn Probability: **{probability:.2f}**')
    if probability > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')
