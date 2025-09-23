import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model1.pkl','rb'))
tfidf = pickle.load(open('vectorizer(max_ft=3000).pkl','rb'))
model = pickle.load(open('model(mnb,max_ft=3000).pkl','rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

st.title("SMS/Email Spam Classifier")

input_msg = st.text_area("Enter the message:")

if st.button('Predict'):

    # Preprocess
    transformed_text = transform_text(input_msg)

    # Vectorise
    vectorised_msg = tfidf.transform([transformed_text])

    # Predict
    prediction = model.predict(vectorised_msg)[0]

    # Display
    if prediction:
        st.header("Spam!")
    else:
        st.header("Not Spam!")
