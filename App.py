import streamlit as st
from ultralytics import YOLO
from PIL import Image

import json

# Model Path
PATH = "best.pt"

st.title("Bahraini Currency Detection")


# Cache resource to avoid continuous loading of model
@st.cache_resource
def load_model(PATH):
    model = YOLO(PATH)
    return model


model = load_model(PATH)

uploaded_file = st.file_uploader(
    "Upload an Image: ", max_upload_size=4, type=[".png", ".jpg"]
)
if uploaded_file is not None:
    if uploaded_file.type in ["image/png", "image/jpeg"]:
        image = Image.open(uploaded_file)

        # Display Result
        prediction = model.predict(image)

        st.write("Detection:")
        for pred in prediction:
            st.image(pred.plot(), width=300)
