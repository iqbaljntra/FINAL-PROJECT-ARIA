import streamlit as st
import pandas as pd
import numpy as np
import joblib

with open('all_process', 'rb') as file_1:
    all_process= joblib.load(file_1)

v1 = st.slider('v1', min_value=280, max_value=600, value=400)
v2 = st.slider('v2', min_value=230, max_value=490, value=350)
v3 = st.slider('v3', min_value=420, max_value=670, value=500)
v4 = st.slider('v4', min_value=300, max_value=450, value=400)
v5 = st.slider('v5', min_value=400, max_value=700, value=550)
v6 = st.slider('v6', min_value=200, max_value=350, value=250)
v7 = st.slider('v7', min_value=450, max_value=900, value=650)
v8 = st.slider('v8', min_value=3800, max_value=4900, value=4350)
sample_type = st.radio('Masukan Jenis Sample', ('lab 1', 'lab 2'))

if st.button('Predict'):
    data = {'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4,
            'v5': v5, 'v6': v6, 'v7': v7, 'v8': v8,
            'sample_type': sample_type}
    df = pd.DataFrame(data, index=[0])
    prediction = all_process.predict(df)
    st.write(f"Prediksi kelas sample adalah {prediction[0]}")
