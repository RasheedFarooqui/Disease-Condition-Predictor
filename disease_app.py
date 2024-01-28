import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import zipfile
import nltk
import os
import streamlit as st
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


rar_archive_path = 'diseasedf.zip'
extract_dir = 'extracted_files'
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(rar_archive_path, 'r') as rar_archive:
    rar_archive.extract('diseasedf.pkl', extract_dir)
    
df = pd.read_pickle(os.path.join(extract_dir, 'diseasedf.pkl'))
# df = pickle.load(open('diseasedf.pkl','rb'))
classifier = pickle.load(open('SVC.pkl','rb'))
vectorizer = pickle.load(open('vectorizerr.pkl','rb'))

print(2)



def predictor(input):
    lemmatizer = WordNetLemmatizer()
    stopword = stopwords.words('english')

    # Text_preprocessing

    temp_corpus = []
    temp_text = []

    text = re.sub(r'[^a-zA-Z]', ' ', input)
    text = text.lower()
    temp_text.append(''.join(text))

    tokens = word_tokenize(temp_text[0])
    tokens = [lemmatizer.lemmatize(word) for word in tokens if not word in set(stopword)]
    temp_corpus.append(' '.join(tokens))

    # TF-IDF
    corpus = vectorizer.transform(temp_corpus)

    # Prediction
    prediction = classifier.predict(corpus)[0]

    #Medication Recommendation
    drugs = []
    dataset = df[df['condition'] == prediction]
    top_drugs = dataset.groupby('drugname')[['rating','usefulcount']].mean().sort_values(ascending=False,by=['rating','usefulcount']).head(5)
    drugs.extend(top_drugs.index.tolist())

    return prediction,drugs


# Title

st.title('Disease Condition Predictor With Recommended Prescriptions')

# Main Function

condition = st.text_area("Enter your disease condition details here:")

if st.button('Predict'):
    if condition == '': 
        st.warning('No input Given')
    else:
        prediction,drugs = predictor(condition)
        st.error(f"Predicted Disease Condition: {prediction}")
        st.write(f"Top Recommended medications: ")
        st.success(", ".join(drugs))


#Drugs Recommender By Rating
def drugs(condition,rating):
    data = df[(df['condition']==condition) & (df['rating']==rating)]
    top = data.sort_values(by='rating', ascending=False).head(10)
    a = top[['drugname','rating']].reset_index(drop=True)
    return a


#Drugs recommender By usefullness & ratings
def drugstop(condition,rating,useful):
    data = df[(df['condition'] == condition) & (df['rating'] == rating)]
    top = data.groupby(['drugname', 'rating'])['usefulcount'].mean().reset_index(name='usefulcount').sort_values(by=['usefulcount','rating'],ascending=False).head(10)
    top = top[top['usefulcount']>useful]
    return top




#Sidebar

tool = st.sidebar.selectbox('Navigation',['Home','Drugs Recommender By Rating','Drugs recommender By usefullness & ratings'])




if tool == 'Drugs Recommender By Rating':

    st.title('Drugs Recommender By Rating')
    condition = st.selectbox('Whats your Disease Condition',df['condition'].unique())
    rating = st.slider('Rating',min_value=5,max_value=10)

    if st.button('Recommend'):
        recommendations = drugs(condition,rating)
        st.write(recommendations[['drugname','rating']],index=False)
        st.write("""_By Rasheed Farooqui._""")



if tool == 'Drugs recommender By usefullness & ratings':
    st.title('Drugs recommender By usefullness & ratings')
    condition = st.selectbox('Whats your Disease Condition',df['condition'].unique())
    rating = st.slider('Rating',min_value=5,max_value=10)
    useful = st.number_input('Usefullness (Based on Average usefullness of a drug on a range from (20-180))',min_value=20,max_value=180)

    if st.button('Recommend'):
        recommendations = drugstop(condition,rating,useful)
        st.write(recommendations)
        st.write("""_By Rasheed Farooqui._""")

##Background
import base64
def add_bg_from_url(image_url):
    response = requests.get(image_url)

    if response.status_code == 200:
        encoded_string = base64.b64encode(response.content)

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_string.decode()});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Failed to download the image from {image_url}")
add_bg_from_url('https://images.unsplash.com/photo-1674702727317-d29b2788dc4a?q=80&w=1973&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')


#Sidebar color

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f0f0; /* Set your desired background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)































