import tensorflow as tf
import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences,  to_categorical

# Main Code
st.title("Simple Sentiment Analysis WebApp") 

text = st.text_area("Please Enter your text :")

# Stopwords
all_stopwords = stopwords.words('english')
remove_words = ["don't" ,'not', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
all_stopwords = set([i for i in all_stopwords if i  not in remove_words])
all_stopwords.add('im')

with open('model_pickel_tokenizer', 'rb') as fe:
    tokenizer = pickle.load(fe)

# def load_my_model():
#     try:
#         model = tf.keras.models.load_model(filepath="LSTM_model.h5")
#     except Exception as e:
#         st.write(e)
#     return model

# model = load_my_model()

model = tf.keras.models.load_model(filepath="LSTM_model.h5")

# Function to preprocess text
def preprocess_text(t):
    text = re.sub('[^a-zA-Z]', ' ', t)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
    text = ' '.join(text)
    return text

# Predict the sentiment
def pred_sentiment(seq):

    word_len = 164 
    
    df_demo = seq.apply(preprocess_text)
    seq_pred = tokenizer.texts_to_sequences(df_demo)
    seq_pred = pad_sequences(seq_pred, maxlen=word_len, padding='post')  # Assuming word_len is defined

    pred = model.predict(seq_pred)

    y_pred = []

    for j in pred:
        i = np.argmax(j)
        if i == 0:
            y_pred.append("Negative")
        elif i == 1:
            y_pred.append("Neutral")
        else:
            y_pred.append("Positive")
            
    return y_pred, pred

# Function to find sentiment
def function(text):

    text_data = {'Text': [text]}
    df_new = pd.DataFrame(text_data)
    df_new['Text'] = df_new['Text'].apply(preprocess_text)
    new_y_pred, pred = pred_sentiment(df_new['Text'])

    st.markdown(f'#### This text is :red[{pred[0][0] * 100 : .2f}% Negative],   :orange[{pred[0][1] * 100 : .2f}% Neutral],   :blue[{pred[0][2] * 100 : .2f}% Positive]')

    if new_y_pred[0] == 'Negative':
        return "The Sentiment is Negative 😭"
    elif new_y_pred[0] == 'Neutral' :
        return "The Sentiment is Neutral 😃"
    elif new_y_pred[0] == 'Positive' :
        return "The Sentiment is Positive 😁"


if st.button("Analyze the Sentiment"):
    result = function(text)
    st.markdown(f'## {result}')