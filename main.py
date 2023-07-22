# Import libraries
from PIL import Image
from io import BytesIO
import base64
import torch
import os
import cv2
import streamlit as st
import torch
from PIL import Image
import numpy as npcla
from torchvision import transforms
import tempfile
import base64
from utility import analyze_image,build_models,create_masked_image,pil_to_byte_array

# Load the page 
st.set_page_config(layout="wide", page_title="Breast lesion genie",page_icon=":female_genie:")


# Page header 
st.markdown("<h1 style='text-align: center;text-shadow: 2px 2px #ffffff; color: pink;'>Mammorgam detection genie</h1>", unsafe_allow_html=True)

# Loader spinner centering only in the first run: 
if 'first_run' not in st.session_state:

    st.session_state['first_run'] = True

    st.markdown("""
    <style>
    div.stSpinner > div {
        text-align:center;
        align-items: center;
        justify-content: center;
    }
    </style>""", unsafe_allow_html=True)

    # Build the models
    with st.spinner('Loading environment...'): 
        st.session_state['model'] = build_models()

# Project theme 
st.markdown(
"""
<div style='text-align: justify; font-family: Arial, sans-serif;'>
<p>This application facilitates the identification, segementation, 
and classification of mammography X-ray images of the breast. 
Its intuitive interface simplifies the process of uploading and analyzing your medical images, thereby augmenting diagnostic precision. 
Initiate the process by uploading a mammogram image.</p>
</div>
""", unsafe_allow_html=True
)

st.sidebar.markdown("## Upload new Mammogram image", unsafe_allow_html=True)

# Function that analyze the image 
def image_analyzer(upload):

    st.markdown("<h2 style='text-align: center; text-shadow: 2px 2px #ffffff; color: pink;'>Analyzed Image</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.2,1,1])

    # Place holder for centering image
    with col1:
        st.write(' ')
    with col3:
        st.write(' ')
    
    image = Image.open(upload)
    resized_image = image.resize((512, 512))  # Resizes the image to be 400 pixels wide and 300 pixels tall

    # Right image - will be a detected lesion according to a map (blue - benign, red - malignant)
    model = st.session_state['model']
    mask = analyze_image(image,model)
    image_with_mask = create_masked_image(image,mask)

    col2.image(image_with_mask.resize((512,512)))

    verdict = 'Malignant'
    # todo - add classification 
    if verdict == 'Malignant':
        col3.markdown("<h2 style='text-align: left; color: #dc3545;'>Malignant</h2>", unsafe_allow_html=True)
    elif verdict == 'Benign':
        col3.markdown("<h2 style='text-align: left; color: #ffc107;'>Benign</h2>", unsafe_allow_html=True)
    else:
        col3.markdown("<h2 style='text-align: left; color: #28a745;'>Healthy</h2>", unsafe_allow_html=True)

    # Add the option to download the image 
    with col3:
        image_bytes = pil_to_byte_array(image_with_mask) 
        col3.download_button(
            label="Download segemented image ",
            data=image_bytes,
            file_name='download.png',
            mime='image/png',
        )
    col3.write(f"Image resulotion is {image_with_mask.size}")


uploaded_file = st.sidebar.file_uploader("Please upload your Mammogram image (Accepted formats: png, jpg, jpeg)", type=["png", "jpg", "jpeg"], help="Drag and drop a file or click to select")

# Analyze one image 
if uploaded_file is not None:

    image_analyzer(uploaded_file)
