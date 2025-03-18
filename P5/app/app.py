# Imports
import pickle
import numpy as np
import uvicorn
import nltk
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
import re
from bs4 import BeautifulSoup
import logging

# Elements necessary for preprocessing
nltk.download('stopwords')
try:
    nlp = spacy.load('en_core_web_sm')
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')


stopwords = nltk.corpus.stopwords.words('english')

def cleaning(text):
    set_stopwords = set(stopwords)

    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    
    doc = nlp(text)
    meaningful_words = [w.lemma_ for w in doc if str(w) not in set_stopwords]   

    return " ".join(meaningful_words)

# FastAPI app setup
app = FastAPI()

# Load your pre-trained models
with open("lda_model.pkl", "rb") as f:
    lda = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("idf_ Attribute Exists:", hasattr(vectorizer, "idf_"))

# Define Pydantic model for input validation
class PredictionInput(BaseModel):
    data: str

from fastapi.responses import JSONResponse

@app.post("/prediction", summary="Retourne une liste des tags prédits")
def get_prediction(input_data: PredictionInput):
    try:
        cleaned_text = cleaning(input_data.data)
        X = vectorizer.transform([cleaned_text]).toarray()
        pred_proba = lda.predict_proba(X)[0]

        # Prepare the response
        res = [
            (label_encoder.inverse_transform([lda.classes_[idx]])[0], prob)
            for idx, prob in enumerate(pred_proba) if prob > 0.1
        ]

        return JSONResponse(content={"predictions": res}, status_code=200)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the app
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=4000, debug=True)
