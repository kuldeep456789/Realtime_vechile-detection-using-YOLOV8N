import streamlit as st
from PIL import Image
import torch
import torch.serialization
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

# Add all potentially needed classes to the safe globals list
torch.serialization.add_safe_globals([
    DetectionModel,
    nn.modules.container.Sequential,
    nn.Sequential,  # Alternative reference to the same class
    nn.ModuleList,
    nn.Module,
    # Add other common PyTorch classes that might be needed
    nn.Conv2d, 
    nn.BatchNorm2d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.Upsample
])

# Set page configuration
st.set_page_config(page_title="Vehicle Tracker", layout="centered")

# App header
st.title("🚗 Real-Time Vehicle Detection with YOLOv8")

@st.cache_resource
def load_model(model_path):
    """Load the YOLO model with explicit weights_only=False to bypass security restrictions."""
    from ultralytics import YOLO
    try:
        # Monkey patch the torch.load function to use weights_only=False
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            # Force weights_only to False for all torch.load calls
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        # Replace torch.load temporarily
        torch.load = patched_load
        
        # Load the model
        model = YOLO(model_path)
        
        # Restore original torch.load
        torch.load = original_load
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Image upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Try to load the model with a context manager
with st.spinner("Loading model..."):
    try:
        # Load the model with a context manager that sets weights_only=False
        with torch.serialization.safe_globals([
            DetectionModel, 
            nn.modules.container.Sequential,
            nn.Sequential,
            nn.ModuleList,
            nn.Module
        ]):
            model = load_model("yolov8n.pt")
            
        if model is None:
            st.error("Could not load the model.")
            st.stop()
            
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Process image if uploaded
if uploaded_file and model:
    # Display loading state
    with st.spinner("Processing image..."):
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Perform detection
            results = model(image)
            
            # Render detection results
            results_img = results[0].plot()
            st.image(results_img, caption="Detected Image", use_container_width=True)
            
            # Display confidence and class information
            st.subheader("Detection Results")
            
            # Get boxes
            boxes = results[0].boxes
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = results[0].names[class_id]
                    
                    st.write(f"Object {i+1}: {class_name} - Confidence: {confidence:.2f}")
            else:
                st.info("No objects detected in the image.")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            import traceback
            st.code(traceback.format_exc())





