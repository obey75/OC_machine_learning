import streamlit as st
import pickle
import re

from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import spacy

nlp = spacy.load("en_core_web_sm")

def cleaning(text):
    set_stopwords = set(stopwords.words("english"))

    text = BeautifulSoup(text).get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    
    doc = nlp(text)
    meaningful_words = [w.lemma_ for w in doc if not str(w) in set_stopwords]   

    return( " ".join(meaningful_words)) 


def predict(input_text):
    cleaned_text = cleaning(input_text)
    
    X = vectorizer.transform([cleaned_text]).toarray()
    pred_proba = model.predict_proba(X)[0]
    
    res = [(label_encoder.inverse_transform([model.classes_[id]])[0],x) for id,x in enumerate(pred_proba) if x>0.1]
    
    return res


# Charger le modèle Hugging Face pré-entraîné
@st.cache_resource
def load_model():
    with open("lda_model.pkl", "rb") as f:
        lda = pickle.load(f)
    
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return lda, label_encoder, vectorizer

model, label_encoder, vectorizer = load_model()


# Interface Streamlit
st.title("Suggestion de tags")

# Description
st.write("""
**Cette application permet de suggérer des tags pertinents pour un post StackOverFlow. 
Le modèle utilisé est basé sur un embedding TF-IDF.
""")

# Entrée utilisateur
input_text = st.text_area("Entrez votre texte :", height=500)

# Prédiction
if st.button("Prédire") and input_text.strip():
    with st.spinner("Analyse en cours..."):
        try:
            # Appeler le modèle pour classer le texte
            cleaned_text = cleaning(input_text)
            res = predict(input_text)


            # Afficher les résultats
            st.subheader("Résultats de la Prédiction :")
            for line in res:  # Parcourir les scores
                label = line[0]
                score = line[1] * 100
                st.write(f"**{label}** : {score:.2f} %")

        except Exception as e:
            st.error(f"Erreur pendant la prédiction : {e}")
else:
    st.info("Entrez du texte et appuyez sur 'Prédire'.")