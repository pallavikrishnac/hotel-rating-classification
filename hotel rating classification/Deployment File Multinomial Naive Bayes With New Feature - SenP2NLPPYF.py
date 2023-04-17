import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as MB
from pickle import dump 
from pickle import load
from keras import models
import pickle
import streamlit as st


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-photo/close-up-raindrops-water-runs-glass-window-pane-surface-grey-blue-sky-background-rainy-day_273651-749.jpg?w=2000");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 



def numm(x):
    if x in (1,2):
         y=1
    elif x==3:
         y=2
    elif x in (4,5):
         y=3
            
    return y  



def split_into_words(i):
    return(i.split(' '))

def calculate_sentiment(text: str =None):
    sent_score = 0
    if text:
        sentence =nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_,0)
    return sent_score

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    text = re.sub('[0-9]" "]+'," ",text)
    text = re.sub('[''""..]','',text)
    return text    



model = pickle.load(open('C:/Users/kkdk0001/Desktop/DA/P2NLPF.sav','rb'))

df = pd.read_excel('C:\\Users\\kkdk0001\\Desktop\\DA\\DS\\Project 2 (P170) -- Hotel Rating Classification\\hotel_reviews.xlsx')
df['Review']= df.Review.apply(clean_text)

dfn=df
dfn['New_Class']=df['Rating'].apply(numm)
dfn=dfn.drop(columns =['Rating'])

review_bow = CountVectorizer(analyzer = split_into_words).fit(dfn['Review'])


def main():
    
    
    
    st.title('Project 2 (P170) -- Hotel Rating Classification')
    st.subheader('Hotel Rating Classification')
    st.sidebar.header('User Input')
    
    x = st.sidebar.text_area('Input Review','Type here')
    test_review_matrix = review_bow.transform([x])
    test_pred_m = model.predict(test_review_matrix)
    
    
    prediction_proba = model.predict_proba(test_review_matrix)
    
    
    
    
    if test_pred_m == 3:
        new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Positive Review</p>'
        new_title_1 = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Customer Would Rate the Hotel with 4 or 5 Star</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.markdown(new_title_1, unsafe_allow_html=True)
        
    elif test_pred_m == 2:
        new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 42px;">Neutral Review</p>'
        new_title_1 = '<p style="font-family:sans-serif; color:Blue; font-size: 42px;">Customer Would Rate the Hotel with 3 Star</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.markdown(new_title_1, unsafe_allow_html=True)
    elif test_pred_m == 1:  
        new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Negative Review</p>'
        new_title_1 = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Customer Would Rate the Hotel with 1 or 2 Star</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.markdown(new_title_1, unsafe_allow_html=True)
        
    st.subheader('Prediction_Probability')
    st.write(prediction_proba)
    
    st.subheader('Prediction_Class')
    st.markdown(test_pred_m)
    
    st.subheader('Review')
    st.markdown(x)
    


if __name__ == '__main__':
  main()