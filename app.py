
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Load Model
# -----------------------------
st.set_page_config(
    page_title="Skin Disease Detection using Deep Learning",
    page_icon="🩺",
    layout="centered"
)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_efficientnetb0.keras")

model = load_model()

# Disease Labels
class_names = [
    "Actinic Keratosis (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevus (nv)",
    "Vascular Lesion (vasc)"
]

# -----------------------------
# Page Title
# -----------------------------
st.title("🩺 Skin Disease Detection")
st.write("Upload a dermoscopic skin image to classify the skin lesion.")
st.sidebar.title("About")

st.sidebar.write("""
This application classifies dermoscopic skin lesion images using
EfficientNetB0 trained on the HAM10000 dataset.

**Model:** EfficientNetB0

**Dataset:** HAM10000

**Classes:** 7 Skin Diseases
""")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224,224))

    img = np.array(img)

    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(f"### Prediction: {class_names[predicted_index]}")

    st.info(f"Confidence: {confidence:.2%}")

    st.progress(float(confidence))

    st.warning(
        "This model is intended for educational purposes only "
        "and should not be used as a substitute for professional medical diagnosis."
    )
    st.subheader("Prediction Probabilities")

    sorted_indices = np.argsort(prediction[0])[::-1]

    for i in sorted_indices:
        st.write(
            f"**{class_names[i]}:** {prediction[0][i]*100:.2f}%"
        )
    st.markdown("---")

    st.caption(
        "Developed using TensorFlow, EfficientNetB0 and the HAM10000 Dataset."
    )       