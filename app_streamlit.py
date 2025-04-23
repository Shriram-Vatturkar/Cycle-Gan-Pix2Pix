import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import time

# Import your model and network definition
from models import networks

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the generator
@st.cache_resource
def load_generator():
    netG = networks.define_G(
        input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks', 
        norm='instance', use_dropout=False, init_type='normal', 
        init_gain=0.02, gpu_ids=[]
    )
    checkpoint = torch.load(
       r"C:\Users\scruk\pytorch-CycleGAN-and-pix2pix\checkpoints\model\latest_net_G_A.pth", map_location=device)
    netG.load_state_dict(checkpoint)
    netG.eval()
    netG.to(device)
    return netG

netG = load_generator()

# Preprocessing function (adjust as needed)
def preprocess(img, load_size=1024, crop_size=1024):
    transform_list = [
        transforms.Resize((load_size, load_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(transform_list)
    return transform(img).unsqueeze(0)

# Postprocessing function
def postprocess(tensor):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = (tensor * 0.5) + 0.5  # [-1,1] to [0,1]
    img = transforms.ToPILImage()(tensor)
    return img

# Streamlit UI
st.title("Style Transfer with CycleGAN")

# --- CycleGAN demonstration placeholder ---
col1, col2, col3 = st.columns([1, 0.2, 1])
with col1:
    st.image(r"C:\Users\scruk\Downloads\WhatsApp Image 2025-04-18 at 12.12.48 PM.jpeg", caption="Image A", use_container_width=True)
with col2:
    st.markdown(
        """
        <div style="display: flex; align-items: center; height: 100%;">
            <h1 style='margin: auto;'>&#8594;</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.image(r"C:\Users\scruk\Downloads\WhatsApp Image 2025-04-18 at 12.12.48 PM (1).jpeg", caption="Image B", use_container_width=True)
# --- End demonstration placeholder ---

st.markdown("<br>", unsafe_allow_html=True)


st.write("Upload an image to apply style transfer your image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Original Image", use_container_width=True)

    with st.spinner("Transferring style..."):
        input_tensor = preprocess(input_image)
        input_tensor = input_tensor.to(device)
        start_time = time.time()  # Start timing
        with torch.no_grad():
            output_tensor = netG(input_tensor)
        inference_time = time.time() - start_time  # End timing
        output_image = postprocess(output_tensor)

    st.image(output_image, caption="Style Transferred Image", use_container_width=True)
    st.write(f"Inference time: {inference_time:.3f} seconds")