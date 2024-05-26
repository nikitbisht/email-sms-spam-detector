import streamlit as st
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
wt = WordNetLemmatizer()

#Data Cleaning
def transform_text(text):
    corpus = []
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    
    text = [wt.lemmatize(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

tfidf = pickle.load(open('models/vectorizer.pkl','rb'))
model = pickle.load(open('models/model.pkl','rb'))

st.title("Email/Sms Spam Classifier")
input_sms = st.text_area("Entere the Message")

if(st.button("Predict")):
    #1 preprocess
    transform_sms = transform_text(input_sms)

    #2vectorize
    vector_input = tfidf.transform([transform_sms])

    #3 predict
    result = model.predict(vector_input)[0]
    if(result==1):
        st.error("Spam")
    else:
        st.success("Not Spam")