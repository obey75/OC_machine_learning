import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import numpy as np
import pickle



# Charger le modèle
#@st.cache(allow_output_mutation=True)
def load_my_model():
    model = load_model('models/vgg_opti.hdf5')
    model.load_weights(f"models/vgg_opti.keras")
    with open("class_indices.pkl", "rb") as f:
        class_indices = list(pickle.load(f))
    return model, class_indices

model, class_indices = load_my_model()

# Prétraiter une image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_preprocessed = preprocess_input(image_array)
    image_batch = np.expand_dims(image_preprocessed, axis=0)
    return image_batch

# Interface Streamlit
st.title("Prédictions de Modèle avec Streamlit")

# Télécharger une ou plusieurs images
uploaded_files = st.file_uploader("Téléchargez des images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Charger l'image
            image = Image.open(uploaded_file)

            # Afficher l'image
            st.image(image, caption=f"Image téléchargée: {uploaded_file.name}", use_container_width=True)

            # Prétraiter et prédire
            image_batch = preprocess_image(image)
            prediction = model.predict(image_batch)
            class_predicted = class_indices[np.argmax(prediction)]
            # Afficher les prédictions
            st.write(f"Prédiction : {class_predicted}")
        except Exception as e:
            st.error(f"Erreur avec le fichier {uploaded_file.name}: {e}")
