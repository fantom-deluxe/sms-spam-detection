import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = []
    for i in tokens:
        if i.isalnum():
            y.append(i)

    tokens = y[:]
    y.clear()

    for i in tokens:
        if i not in stopwords.words('english'):
            y.append(i)

    tokens = y[:]
    y.clear()

    for i in tokens:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # preprocessing
    transformed_sms = transform_text(input_sms) 

    # vectorize
    vector_input = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vector_input)

    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")



