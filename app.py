import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tempfile
from streamlit_option_menu import option_menu

# Constants
IMG_WIDTH = 150
IMG_HEIGHT = 150
MODEL_PATH = "malaria_AI_model.h5"

# Load the model
@st.cache_resource
def load_malaria_model():
    model = load_model(MODEL_PATH)
    return model

# Function to evaluate the uploaded image
def evaluate_user_image(image_path, model, img_width, img_height):
    """
    Evaluates a single image provided by the user.

    Parameters:
        image_path (str): Path to the image file.
        model (keras.Model): Trained model for prediction.
        img_width (int): Target width for resizing the image.
        img_height (int): Target height for resizing the image.
    """
    image = load_img(image_path, target_size=(img_width, img_height))
    img_arr = img_to_array(image) / 255.0
    pred = model.predict(img_arr.reshape(1, *img_arr.shape), verbose=0).flatten()
    label = "Parasitised" if pred < 0.5 else "Uninfected"
    return label, img_arr

# Function to evaluate multiple images
def evaluate_multiple_images(image_paths, model, img_width, img_height):
    results = []
    for image_path in image_paths:
        label, img_arr = evaluate_user_image(image_path, model, img_width, img_height)
        results.append((label, img_arr))
    return results

# Set the page configuration
st.set_page_config(
    page_title="Malaria AI-Classifier",
    page_icon="🦠",
    layout="wide",
)

# Apply custom CSS for styling
def set_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #bed8ff;
        }
        .css-1d391kg {
            background-color: #707ff5;
        }
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1lcbmhc {
            color: #68a2f1;
        }
        div.stButton > button {
            background-color: #005f99;
            color: white;
            font-size: 14px;
            border-radius: 8px;
        }
        div.stButton > button:hover {
            background-color: #004080;
            color: white;
        }
        .css-18ni7ap {
            background-color: #a195f9;
            border-radius: 8px;
        }
        .css-1y4p8pa .css-1d391kg {
            color: #68a2f1;
        }
        h1 {
            font-size: 26px;
            font-weight: bold;
            color: #333333;
        }
        h2 {
            font-size: 20px;
            font-weight: bold;
            color: #444444;
        }
        h3 {
            font-size: 18px;
            font-weight: bold;
            color: #555555;
        }
        p, li {
            font-size: 14px;
            color: #666666;
        }
        .stImage {
            max-width: 100%;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Main function for the Streamlit app
def main():
    set_custom_styles()

    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Home", "Upload Image", "Multiple Image Upload", "About"],
            icons=["house-door", "cloud-upload", "images", "info-circle"],
            menu_icon="list",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#f0f0f0"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#d1e8fc",
                },
                "nav-link-selected": {"background-color": "#68a2f1", "color": "white"},
            },
        )

    model = load_malaria_model()

    if selected == "Home":
        st.title("Malaria AI-Classifier")
        st.write(
            """
            ### Malaria Life Cycle
            Malaria is caused by *Plasmodium* parasites and transmitted by Anopheles mosquitoes. Key stages include:
            1. **Sporozoite**: Parasites enter the liver via the bloodstream.
            2. **Liver Stage**: Parasites multiply in liver cells.
            3. **Blood Stage**: Parasites infect red blood cells, causing symptoms.

            Below is a visual of the malaria life cycle:
            """
        )
        st.image(
            "https://www.cdc.gov/dpdx/malaria/modules/malaria_LifeCycle.gif?_=05237",
            caption="Malaria Life Cycle (Source: CDC)",
            use_container_width=True,
        )

        st.write(
            """
            ### Detection Methods
            - **Microscopy**: Observing stained blood smears.
            - **Rapid Diagnostic Tests**: Detecting antigens in the blood.
            - **AI Models**: Automated detection through image analysis.
            """
        )
        st.image(
            "https://debuggercafe.com/wp-content/uploads/2023/08/malaria-classification-vision-transformer-ground-truth-images.png",
            caption="AI Model Predictions (Source: DebuggerCafe)",
            use_container_width=True,
        )

    elif selected == "Upload Image":
        st.title("AI Classifier - Upload Image")

        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_image_path = tmp_file.name

            label, img_arr = evaluate_user_image(tmp_image_path, model, IMG_WIDTH, IMG_HEIGHT)

            col1, col2 = st.columns(2)

            with col1:
                st.image(img_arr, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.success(f"Prediction: {label}")

    elif selected == "Multiple Image Upload":
        st.title("AI Classifier - Multiple Image Upload")

        uploaded_files = st.file_uploader(
            "Choose multiple image files", type=["jpg", "png", "jpeg"], accept_multiple_files=True
        )

        if uploaded_files:
            tmp_image_paths = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_image_paths.append(tmp_file.name)

            results = evaluate_multiple_images(tmp_image_paths, model, IMG_WIDTH, IMG_HEIGHT)

            for i, (label, img_arr) in enumerate(results):
                st.write(f"### Image {i + 1}")
                col1, col2 = st.columns(2)

                with col1:
                    st.image(img_arr, caption=f"Uploaded Image {i + 1}", use_container_width=True)

                with col2:
                    st.success(f"Prediction: {label}")

    elif selected == "About":
        st.title("About this App")
        st.write(
            """
            This **Malaria Classifier** uses deep learning to analyze blood smear images and diagnose malaria.

            - **Model**: VGG19-based with custom layers.
            - **Input Size**: 150x150 pixels.

            Created by **Dr. Majeed Jamakhani**.
            """
        )

if __name__ == "__main__":
    main()
