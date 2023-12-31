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
import numpy as np
from torchvision import transforms
import tempfile
import base64
from utility import analyze_image,build_models,create_masked_image,pil_to_byte_array,model_classification

# Load the page 
st.set_page_config(layout="wide", page_title="Breast lesion genie",page_icon=":female_genie:")


# Page header - welcome page (before the selection of an option)
if 'selection' not in st.session_state or st.session_state['selection']== 'None':
    st.markdown("<h1 style='text-align: center;text-shadow: 2px 2px #ffffff; color: pink;'>Mammorgam detection genie</h1>", unsafe_allow_html=True)
elif st.session_state['selection'] == 'Investigate a mammogram':
    st.markdown("<h1 style='text-align: center;text-shadow: 2px 2px #ffffff; color: pink;'>Investigate a mammogram</h1>", unsafe_allow_html=True)
elif st.session_state['selection'] == 'Train the model':
    st.markdown("<h1 style='text-align: center;text-shadow: 2px 2px #ffffff; color: pink;'>Train the model</h1>", unsafe_allow_html=True)

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



# Select one of the options: 
# 1. Train the model 
# 2. Inspect mammogram 
# Project theme 
if 'selection' not in st.session_state or st.session_state['selection']== 'None':
    
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
    
    # Options selection
    selected_option = st.selectbox("Select one of the options:",options = ('select...','How to train the model', 'Investigate a mammogram'))

    # Selected to Train the model 
    if selected_option == 'How to train the model':
        st.session_state['selection'] = 'How to train the model'
        # Run again to remove the headline
        st.experimental_rerun()
    
    # Selected to investigate a mammogram
    elif selected_option == 'Investigate a mammogram':
        st.session_state['selection'] = 'Investigate a mammogram'
        # Run again to remove the headline
        st.experimental_rerun()
    else:
        st.session_state['selection'] = 'None'

####################################################################
###################### Option 1 ####################################
####################################################################

#  Option #1 selected: Train the model
if st.session_state['selection'] == 'How to train the model':
    
    # back button functionality 
    # Create columns. The first one takes up most of the space, the second one is for the button.
    col1, col2 = st.columns([4,1])

    with col1:
        st.write("")  # Just to put something here.

    with col2:
        if st.button('Go back to Main menu :back:'):
            st.session_state['selection'] = 'None'
            # Run again to remove the headline
            st.experimental_rerun()
    
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["1\. About the Dataset", "2\. Train the YOLO Model", 
                                      "3\. Train the patch Model","4\. Train the classifier Model","5\. Evaluate the Full model"])

    # 1. About the dataset 
    with tab1:
        st.markdown(
        """
        <div style='text-align: justify; font-family: Arial, sans-serif;'>
        <p>For training the model,You will need to follow these guidlines:
        \n ## Downloading the resources
        \nDownload the images folder from the following link: https://drive.google.com/drive/folders/1R4oc9YbQeK3cofilDLNbmEvuGCMczPtC?usp=sharing
        \nCode for all is available on : https://github.com/amirsalhuv/MassDetector

        \n ## Prepration of your own data 
        \n1. For adding your own data, you will need to create pairs of data for image with its corresponding mask. The numbering starts from last number in table.csv including leading zeros.
        \n2. Add your new images to the table.csv. __Manadatory__ fields are: image_id (image/mask name) and pathology(case sensitive). All the rest are recommneded.
        \n3. The Images should be from shape [H,W,3] . Masks are of shape [H,W] and normlized to 1 
        \n4. For labels files creation go to "Train the YOLO Model" tab
        \n5. Select your own division to training and validation, curret division is 0.8/0.2 for training/validation 
        \n## Folders structure</p></div>
        """, unsafe_allow_html=True
        )
        st.image("folders.png",caption='Dataset folder',width=1000)
        st.markdown(
        """
        <div style='text-align: justify; font-family: Arial, sans-serif;'><p>
        \n ## Training process \n </p>
        </div>
        """, unsafe_allow_html=True
        )
        st.image("Training_process.png",caption='Training block diagram',width=1000)

    # 2. Train the YOLO model 
    with tab2:
        st.markdown(
        """
        <div style='text-align: justify; font-family: Arial, sans-serif;'>
        <p>For training the YOLO model, follow these steps:
        \n 1. YOLO code is availble on the following link: https://github.com/amirsalhuv/MassDetector/blob/main/pytorch_model_YOLOv8Model.ipynb
        \n 2. For creating the YOLO label folder, you can run the "Create the Yolo folder" block after adding your data.For running, change the run mode to run_mode = "create_yolo_folder" 
        \n 3. For training the model, modify the run_mode to run_mode="training_model"
        \n 4. Training parameters can be modified in line 20 of "Training the model" block:
        \n      !python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --cfg yolov5x.yaml --weights yolov5x.pt --project training_folder+"yolov5_models"
        \n *"img"*: image size to convert to (currently 640x640)
        \n *"batch"*: batch size (currently 16)
        \n *"epochs"*: max number of epocs (currently 100) 
        \n *"Training folder"*: Folder where the model will be saved 
        </p>
        </div>
        """, unsafe_allow_html=True
        )

    # 3. Train the Patch 
    with tab3:
        st.markdown(
        """
        <div style='text-align: justify; font-family: Arial, sans-serif;'>
        <p>For training the Patch model, follow these steps:
        \n 1. Patch model is availble on the following link: https://github.com/amirsalhuv/MassDetector/blob/main/IOU_loss_patch_model_2_classes_.ipynb
        \n 2. Change the dataset_dir to your own folder 
        \n 3. **Create patchs:** modify the run mode to run_mode = "cut_pathces" and run the script 
        \n 4. **Training model:** Train the model to run_mode = "training_model"
        \n 5. Before running the trainning section, you can modify a couple of hyper parameters in the "Defintions" block:
        </p>
        </div>
        """, unsafe_allow_html=True
        )
        st.image("Patch_model_defintions.png",width=1000)
        st.markdown(
        """
        <div style='text-align: justify; font-family: Arial, sans-serif;'>
        <p>
        \n 1. "n_classes" - number of classes, should remain 2 
        \n 2. "batch_size" - batch size 
        \n 3. "lr" - initial learning rate
        \n 4. "NUM_OF_EPOCHS_TO_REDUCE_LR" - number of consequitive epochs w/o imporvment that will cause a reduction of the learning rate by a factor of 10 
        \n 5. "NUM_OF_EPOCHS_TO_STOP" - number of consequitive epochs w/o imporvment that will cause a stop of training. usually will be larger than NUM_OF_EPOCHS_TO_REDUCE_LR
        </p>
        </div>
        """, unsafe_allow_html=True
        )

    # 4. Train the Classifer 
    with tab4:
        st.markdown(
        """
        <div style='text-align: justify; font-family: Arial, sans-serif;'>
        <p>For training the Classifer model, follow these steps:
        \n 1. Classifer model is availble on the following link: https://github.com/amirsalhuv/MassDetector/blob/main/Classfication_model.ipynb 
        \n 2. Change the dataset_dir to your own folder 
        \n 3. **Training model:** Train the model to run_mode = "training_model"
        \n 4. Before running the trainning section, you can modify a couple of hyper parameters in the "Defintions" block:
        </p>
        </div>
        """, unsafe_allow_html=True
        )
        st.image("Patch_model_defintions.png",width=1000)
        st.markdown(
        """
        <div style='text-align: justify; font-family: Arial, sans-serif;'>
        <p>
        \n 1. "n_classes" - number of classes, should remain 2 
        \n 2. "batch_size" - batch size 
        \n 3. "lr" - initial learning rate
        \n 4. "NUM_OF_EPOCHS_TO_REDUCE_LR" - number of consequitive epochs w/o imporvment that will cause a reduction of the learning rate by a factor of 10 
        \n 5. "NUM_OF_EPOCHS_TO_STOP" - number of consequitive epochs w/o imporvment that will cause a stop of training. usually will be larger than NUM_OF_EPOCHS_TO_REDUCE_LR
        </p>
        </div>
        """, unsafe_allow_html=True
        )

    # 4. Evaluate the Full model
    with tab5:
        st.markdown(
        """
        <div style='text-align: justify; font-family: Arial, sans-serif;'>
        <p>For Evaluating the Full model, follow these steps:
        \n 1. Download the vs code from this repository https://github.com/amirsalhuv/MassDetector/blob/main/Final_model.ipynb
        \n 2. Change the dataset_dir to your own folder 
        \n 3. Make sure your .npy files are availible
        \n 4. Make sure your run_mode is set to run_mode = "evaluate_model"
        \n 5. Enjoy the ride
        \n ## Evaluation process</p></div>
        """, unsafe_allow_html=True
        )
        st.image("evaluating_the_model.png",width=1000)

####################################################################
###################### Option 2 ####################################
####################################################################
#  Option #2 selected: Investigate a mammogram
if st.session_state['selection'] == 'Investigate a mammogram':
    
    # back button functionality 
    # Create columns. The first one takes up most of the space, the second one is for the button.
    col1, col2 = st.columns([4,1])

    with col1:
        st.write("")  # Just to put something here.

    with col2:
        if st.button('Go back to Main menu :back:'):
            st.session_state['selection'] = 'None'
            # Run again to remove the headline
            st.experimental_rerun()

    # Create the side bar for uploading an image 
    st.sidebar.markdown("## Upload new Mammogram image", unsafe_allow_html=True)

    # Function that analyze the image 
    def image_analyzer(upload):

        st.markdown("<h2 style='text-align: center;color:white;'>Analyzed Image</h2>", unsafe_allow_html=True)

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
        image_with_mask,largest_contour_area = create_masked_image(image,mask)

        col2.image(image_with_mask.resize((512,512)))

        # Classification 
        verdict = model_classification(image,mask)

        # Seperators 
        col3.markdown("<h2 style='text-align: left; color: white;'>Lesion info</h2>", unsafe_allow_html=True)

        # Chcek what is the verdict 
        if verdict == 'Malignant':
            col3.markdown("Mammogram has **:red[:warning: Malignant] lesions**")
        elif verdict == 'Benign':
            col3.markdown("Mammogram has **:orange[:warning: Benign] lesions**")
        else:
            col3.markdown("**:green[:heavy_check_mark: No lesion ]**")

        col3.write(f"Image resulotion: {image_with_mask.size} $pixels$")
        col3.write(f"Largest lesion area: {largest_contour_area} $$pixels^2$$ ")

        # Add the option to download the image 
        with col3:
            image_bytes = pil_to_byte_array(image_with_mask) 
            col3.download_button(
                label="Download segemented image ",
                data=image_bytes,
                file_name='download.png',
                mime='image/png',
            )


    uploaded_file = st.sidebar.file_uploader("Please upload your Mammogram image (Accepted formats: png, jpg, jpeg)", type=["png", "jpg", "jpeg","pgm"], help="Drag and drop a file or click to select")

    # Analyze one image 
    if uploaded_file is not None:

        image_analyzer(uploaded_file)
