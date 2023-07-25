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
from torchvision import transforms

min_roi_size = 100
padding = 200

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

    if contours:
         # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
         # Compute the area of the largest contour
        largest_contour_area = cv2.contourArea(largest_contour)
        # Draw all contours 
        for cnt in contours:
            # Draw the largest contour on the image
            image_np = cv2.drawContours(image_np, [cnt], -1, (255,255,0), 10)
    
    else: # No contour detected
        largest_contour_area = 0

    # Convert the NumPy array with the contour back to a PIL Image
    image_with_contour = Image.fromarray(image_np)

    return image_with_contour,largest_contour_area


def pil_to_byte_array(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    return byte_arr.getvalue()



def model_classification(image,mask):
    
    classifier = class_models.Classifer_model()
    # Load the model weights
    classifier.load_state_dict(torch.load('classifier_model_padding.pth',map_location=torch.device('cpu')))
    # Move to evaluation:
    classifier.eval()

    # Create the transform before inputing 
    test_image_transforms = transforms.Compose([
              transforms.Resize((224, 224)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ] )

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
    
    lesion = "Healthy"
    with torch.no_grad():
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h > min_roi_size:
                # Update the input according
                img_np =cv2.resize(image_np[max(0,y-padding):min(mask_np.shape[0]-1,y+h+padding),max(0,x-padding):min(mask_np.shape[1]-1,x+w+padding),:],(256,256))
                
                # Convert back to PIL for the transformations 
                img = Image.fromarray(img_np.astype('uint8'))

                # Build the transform before the model 
                img = test_image_transforms(img).unsqueeze(0)

                # run the model 
                output = classifier(img)

                _, preds = torch.max(output, 1)

                if preds==1:
                    # In case we detected a malignant lesion, no point continue 
                    lesion = "Malignant"
                    return lesion
                else:
                    # If not a maliganant, it is benign
                    lesion = "Benign"
        
        return lesion












