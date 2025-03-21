import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Chargement du modèle CNN (version améliorée)
model = tf.keras.models.load_model("mon_cnn_amelioré.h5")  # Nom du modèle mis à jour
IMG_SIZE = (224, 224)  # Doit MATCHER la taille d'entraînement

# Vérifier l'ordre des classes selon votre dataset
labels = {0: "Conforme", 1: "Non conforme"}  # Ordre à confirmer

def main():
    st.title("Contrôle Qualité : Vis")
    st.write("OkVisFactory - Diagnostic de qualité automatisé")

    uploaded_file = st.file_uploader("Chargez une image de vis (.jpg/.png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Chargement et conversion RGB
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Vis analysée", use_column_width=True)

            # Prétraitement optimisé
            image_resized = image.resize(IMG_SIZE)
            image_array = np.array(image_resized, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Prédiction avec vérification
            if model.input_shape[1:3] != IMG_SIZE:
                raise ValueError(f"Taille d'image incompatible. Attendu : {model.input_shape[1:3]}")

            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100

            # Affichage professionnel
            st.markdown(f"**Résultat :** `{labels[predicted_class]}`")
            st.metric("Confiance du modèle", f"{confidence:.2f}%")

            if labels[predicted_class] == "Conforme":
                st.success("✅ Vis conforme aux standards qualité")
            else:
                st.error("🚨 Défaut critique détecté - Rejet automatique")

        except Exception as e:
            st.error(f"ERREUR : {str(e)}")
            st.info("Astuce : Utilisez des images 224x224 pixels en format RGB")

if __name__ == "__main__":
    main()