from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from plant_classifier.model_store import load_model_bundle
from plant_classifier.inference import predict_uploaded_image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "full_random_forest.pickle"


st.set_page_config(
    page_title="Plant Leaf Disease Classification",
    page_icon="🌿",
    layout="wide",
)


@st.cache_resource
def load_model():
    return load_model_bundle(MODEL_PATH)


def format_confidence(confidence):
    if confidence is None:
        return "Not available"
    return f"{confidence * 100:.2f}%"


st.title("🌿 Plant Leaf Disease Classification")
st.write(
    "Upload a plant leaf image to classify it into one of the selected healthy or diseased classes. "
    "This MVP uses handcrafted Computer Vision features and a Random Forest model."
)

with st.expander("Supported classes"):
    st.write(
        """
        - Peach - Bacterial spot
        - Peach - Healthy
        - Pepper bell - Bacterial spot
        - Pepper bell - Healthy
        - Strawberry - Leaf scorch
        - Strawberry - Healthy
        """
    )

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is None:
    st.info("Upload an image to start prediction.")
    st.stop()

try:
    model_bundle = load_model()
except Exception as error:
    st.error("Failed to load model.")
    st.exception(error)
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    result = predict_uploaded_image(image_rgb, model_bundle)

    predicted_label = result["predicted_label"]
    confidence = result["confidence"]
    features = result["features"]

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Class", predicted_label)

    with col2:
        st.metric("Confidence", format_confidence(confidence))

    with col3:
        st.metric("Disease Area Ratio", f"{features['disease_area_ratio']:.4f}")

    st.subheader("Image Visualization")

    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.image(result["processed_rgb"], caption="Uploaded Image", use_container_width=True)

    with img_col2:
        st.image(result["spot_mask"], caption="Disease Spot Mask", use_container_width=True)

    with img_col3:
        st.image(result["overlay"], caption="Disease Spot Overlay", use_container_width=True)

    st.subheader("Extracted Feature Summary")

    selected_features = {
        "disease_area_ratio": features.get("disease_area_ratio"),
        "spot_count": features.get("spot_count"),
        "largest_spot_ratio": features.get("largest_spot_ratio"),
        "glcm_contrast": features.get("glcm_contrast"),
        "glcm_homogeneity": features.get("glcm_homogeneity"),
        "glcm_energy": features.get("glcm_energy"),
    }

    st.dataframe(
        pd.DataFrame([selected_features]).T.rename(columns={0: "value"}),
        use_container_width=True,
    )

except Exception as error:
    st.error("Prediction failed. Please try another image.")
    st.exception(error)