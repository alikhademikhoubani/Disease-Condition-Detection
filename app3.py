import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Import the model
model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
count_vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

st.title('Disease Condition Detection')

# Input Text
input_text = st.text_area('Input Review')

if st.button('Detect'):
    # query
    query = np.array([input_text])
    count_text = count_vectorizer.transform(query)

    predict = model.predict(count_text)
    if predict == 0:
        b = 'Birth Control'
    elif predict == 1:
        b = 'Depression'
    elif predict == 2:
        b = 'Diabetes, Type 2'
    else:
        b = 'High Blood Pressure'
    st.title('This review is related to ' + b + ' ' + 'Disease')