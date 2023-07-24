import torch 
from torchvision import models
import ssl
import psutil
import ultralytics
import time
import streamlit as st
import class_models
import numpy as np
from PIL import Image
import cv2
import io



def analyze_image(image,model):
    with st.spinner('Analyzing model...'):
        
        # Preprocess the image: convert to RGB and change to numpy array
        image = image.convert('RGB') 
        image = np.array(image) 

        # Analyze the model 
        mask = model(image)

        # Convert:
        # 1. back to [0-255] from [0-1] 
        # 2. And Covert back to Pil
        mask = Image.fromarray(mask*255)

    return(mask)


def build_models():
    
    # import YOLOv5 model
    ssl._create_default_https_context = ssl._create_unverified_context
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo.pt')

    # Import patch model 
    ##### Load Patch model  ######
    patch_model = class_models.VGGUnet(models.vgg16_bn, pretrained=True, out_channels=1)
    # Load the model weights
    patch_model.load_state_dict(torch.load('Patch_model_padding_2_classes.pth',map_location=torch.device('cpu')))

    # Create the combined model 
    model = class_models.CombinedModel(yolo_model,patch_model)
    
    # Move into evaluation 
    model.eval()

    return model


def create_masked_image(image, mask):
    # Ensure image is in RGB mode
    image = image.convert("RGB")

    # Ensure mask is in L mode
    mask = mask.convert("L")

    # Convert PIL images to NumPy arrays for OpenCV
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Find contours in the mask. This returns a list of contours, where each contour
    # is an array of points. We only keep the largest contour.
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the area of the largest contour
    largest_contour_area = cv2.contourArea(largest_contour)

    # Draw the largest contour on the image
    image_np = cv2.drawContours(image_np, [largest_contour], -1, (255,255,0), 10)

    # Convert the NumPy array with the contour back to a PIL Image
    image_with_contour = Image.fromarray(image_np)

    return image_with_contour,largest_contour_area


def pil_to_byte_array(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    return byte_arr.getvalue()












