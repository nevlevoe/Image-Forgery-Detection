# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:38:36 2025

@author: msada
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import io


# ========== Setup and Imports from User Models ==========
from resnet50_run import ForgeryResNet, convert_to_ela_image, eval_transform
from mvssnetrun import get_mvss
from deepfake_densenet import load_deepfake_model , classify_deepfake


# ========== Configurations ==========
RESNET_PATH = "./best_combined_resnet_model.pth"
MVSSNET_PATH_HIGH = "./defactomvssnet.pt"
MVSSNET_PATH_LOW = "./casiamvssnet.pt"
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load Models ==========
@st.cache_resource
def load_resnet():
    model = ForgeryResNet().to(DEVICE)
    model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
def load_mvssnet(model_path):
    model = get_mvss(backbone='resnet50',
                     pretrained_base=True,
                     nclass=1,
                     sobel=True,
                     constrain=True,
                     n_input=3)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model

# ========== Prediction Pipeline ==========
def classify_image(pil_image: Image.Image, threshold=0.85):
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    ela_image = convert_to_ela_image(buffer)
    input_tensor = eval_transform(ela_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        prob = resnet_model(input_tensor).item()
        label = "Forged" if prob > 0.5 else "Authentic"
    return label, prob


def segment_with_mvssnet(image: Image.Image, model):
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        edge_map, mask = model(input_tensor)
    
    mask = mask.squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)
    return mask, binary_mask

# ========== Streamlit UI ==========
st.title("🔍 Image Forgery Detection")

uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ========== Step 1: DeepFake Check First ==========
    deepfake_model = load_deepfake_model()
    df_label, df_confidence = classify_deepfake(image, deepfake_model)
    print(f"** DF Confidence:** {df_confidence:.4f}")
    
    if df_label == "Fake":
        st.markdown(f"### Forgery Detection Classification Result: **{df_label}**")
        st.warning("Model predicts this image as **Fake**.")
    
    else:
        resnet_model = load_resnet()
        label, confidence = classify_image(image)

        st.markdown(f"### Forgery Detection Classification Result: **{label}**")
        print(f"** TD Confidence:** {confidence:.4f}")
        # ========== Step 2: Proceed only if DeepFake is Real ==========
        if label == "Forged":
            st.warning("Model predicts this image as **Fake**.")
            st.success("Showing Masks for both low and high resolution image models: ")
    
            # Load both models
            mvss_model_casia = load_mvssnet(MVSSNET_PATH_LOW)
            mvss_model_defacto = load_mvssnet(MVSSNET_PATH_HIGH)
    
            # Run segmentation with both
            mask_casia, binary_mask_casia = segment_with_mvssnet(image, mvss_model_casia)
            mask_defacto, binary_mask_defacto = segment_with_mvssnet(image, mvss_model_defacto)
    
            st.markdown("### 🔍 Forgery Segmentation Results ")
    
            # Display side-by-side
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
            # CASIA model results
            axs[0][0].imshow(mask_casia, cmap='gray', vmin=0, vmax=1)
            axs[0][0].set_title("low resolution image model Predicted Mask")
            axs[0][0].axis('off')
    
            axs[0][1].imshow(binary_mask_casia * 255, cmap='gray', vmin=0, vmax=255)
            axs[0][1].set_title("low resolution image Binarized Mask")
            axs[0][1].axis('off')
    
            # DEFACTO model results
            axs[1][0].imshow(mask_defacto, cmap='gray', vmin=0, vmax=1)
            axs[1][0].set_title("High resolution image model Predicted Mask")
            axs[1][0].axis('off')
    
            axs[1][1].imshow(binary_mask_defacto * 255, cmap='gray', vmin=0, vmax=255)
            axs[1][1].set_title("High resolution image model Binarized Mask")
            axs[1][1].axis('off')
    
            st.pyplot(fig)
    
        else:
            st.info("✅ The image is predicted as **authentic** by  forgery model. No manipulation detected.")


