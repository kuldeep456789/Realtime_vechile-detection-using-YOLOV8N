# import streamlit as st
# from PIL import Image
# import torch
# import torch.serialization
# from ultralytics import YOLO
# import torch.nn as nn
# from ultralytics.nn.tasks import DetectionModel
# import tempfile
# import cv2
# import numpy as np

# # Add necessary classes to safe globals
# torch.serialization.add_safe_globals([
#     DetectionModel,
#     nn.modules.container.Sequential,
#     nn.Sequential,
#     nn.ModuleList,
#     nn.Module,
#     nn.Conv2d,
#     nn.BatchNorm2d,
#     nn.ReLU,
#     nn.LeakyReLU,
#     nn.Upsample
# ])

# # Streamlit config
# st.set_page_config(page_title="Vehicle Tracker", layout="centered")
# st.title("🚗 Real-Time Vehicle Detection with YOLOv8")

# @st.cache_resource
# def load_model(model_path):
#     try:
#         original_load = torch.load

#         def patched_load(*args, **kwargs):
#             kwargs['weights_only'] = False
#             return original_load(*args, **kwargs)

#         torch.load = patched_load
#         model = YOLO(model_path)
#         torch.load = original_load
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Upload file (image or video)
# file_type = st.radio("Select input type:", ("Image", "Video"))
# uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "mp4"])

# # Load model
# with st.spinner("Loading model..."):
#     try:
#         with torch.serialization.safe_globals([
#             DetectionModel,
#             nn.modules.container.Sequential,
#             nn.Sequential,
#             nn.ModuleList,
#             nn.Module
#         ]):
#             model = load_model("yolov8n.pt")

#         if model is None:
#             st.stop()
#     except Exception as e:
#         st.error(f"Failed to load model: {str(e)}")
#         st.stop()

# # Process image
# if uploaded_file and model and file_type == "Image":
#     with st.spinner("Processing image..."):
#         try:
#             image = Image.open(uploaded_file)
#             st.image(image, caption="Uploaded Image", use_container_width=True)

#             results = model(image)
#             results_img = results[0].plot()
#             st.image(results_img, caption="Detected Image", use_container_width=True)

#             st.subheader("Detection Results")
#             boxes = results[0].boxes

#             if len(boxes) > 0:
#                 for i, box in enumerate(boxes):
#                     confidence = box.conf.item()
#                     class_id = int(box.cls.item())
#                     class_name = results[0].names[class_id]
#                     st.write(f"Object {i+1}: {class_name} - Confidence: {confidence:.2f}")
#             else:
#                 st.info("No objects detected.")
#         except Exception as e:
#             st.error(f"Error processing image: {str(e)}")

# # Process video
# elif uploaded_file and model and file_type == "Video":
#     with st.spinner("Processing video..."):
#         try:
#             # Save the uploaded video to a temp file
#             temp_file = tempfile.NamedTemporaryFile(delete=False)
#             temp_file.write(uploaded_file.read())
#             temp_file_path = temp_file.name

#             cap = cv2.VideoCapture(temp_file_path)
#             stframe = st.empty()

#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = model(frame_rgb)
#                 result_frame = results[0].plot()
#                 stframe.image(result_frame, channels="RGB", use_container_width=True)

#             cap.release()

#         except Exception as e:
#             st.error(f"Error processing video: {str(e)}")



import streamlit as st
from PIL import Image
import torch
import torch.serialization
from ultralytics import YOLO
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
import tempfile
import cv2
import numpy as np
import pandas as pd
import altair as alt

# Add necessary classes to safe globals
torch.serialization.add_safe_globals([
    DetectionModel,
    nn.modules.container.Sequential,
    nn.Sequential,
    nn.ModuleList,
    nn.Module,
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.Upsample
])

# Streamlit config
st.set_page_config(page_title="Vehicle Tracker", layout="centered")
st.title("🚗 Real-Time Vehicle Detection and Counting with YOLOv8")

@st.cache_resource
def load_model(model_path):
    try:
        original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load
        model = YOLO(model_path)
        torch.load = original_load
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize session state for vehicle counts
if 'vehicle_counts' not in st.session_state:
    st.session_state['vehicle_counts'] = {}

# Initialize session state for total vehicles passed
if 'total_vehicles_passed' not in st.session_state:
    st.session_state['total_vehicles_passed'] = 0

# Upload file (image or video)
file_type = st.radio("Select input type:", ("Image", "Video"))
uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "mp4"])

# Load model
with st.spinner("Loading model..."):
    try:
        with torch.serialization.safe_globals([
            DetectionModel,
            nn.modules.container.Sequential,
            nn.Sequential,
            nn.ModuleList,
            nn.Module
        ]):
            model = load_model("yolov8n.pt")

        if model is None:
            st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Function to update vehicle counts
def update_vehicle_counts(results):
    detected_classes = results[0].names
    boxes = results[0].boxes
    current_frame_counts = {}
    if len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls.item())
            class_name = detected_classes[class_id]
            current_frame_counts[class_name] = current_frame_counts.get(class_name, 0) + 1

    for vehicle_type, count in current_frame_counts.items():
        st.session_state['vehicle_counts'][vehicle_type] = st.session_state['vehicle_counts'].get(vehicle_type, 0) + count
        st.session_state['total_vehicles_passed'] += count

# Process image
if uploaded_file and model and file_type == "Image":
    with st.spinner("Processing image..."):
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            results = model(image)
            results_img = results[0].plot()
            st.image(results_img, caption="Detected Image", use_container_width=True)

            st.subheader("Detection Results")
            boxes = results[0].boxes

            if len(boxes) > 0:
                update_vehicle_counts(results)
                for i, box in enumerate(boxes):
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = results[0].names[class_id]
                    st.write(f"Object {i+1}: {class_name} - Confidence: {confidence:.2f}")
            else:
                st.info("No objects detected.")

            st.subheader("Vehicle Count Summary")
            st.write(f"Total Vehicles Passed: {st.session_state['total_vehicles_passed']}")
            if st.session_state['vehicle_counts']:
                df = pd.DataFrame(list(st.session_state['vehicle_counts'].items()), columns=['Vehicle Type', 'Count'])
                chart = alt.Chart(df).mark_bar().encode(
                    x='Vehicle Type',
                    y='Count',
                    tooltip=['Vehicle Type', 'Count']
                ).properties(
                    title='Number of Vehicles Passed by Type'
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No vehicles detected yet to show summary.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Process video
elif uploaded_file and model and file_type == "Video":
    with st.spinner("Processing video..."):
        try:
            # Save the uploaded video to a temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

            cap = cv2.VideoCapture(temp_file_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                result_frame = results[0].plot()
                stframe.image(result_frame, channels="RGB", use_container_width=True)

                update_vehicle_counts(results)

            cap.release()

            st.subheader("Vehicle Count Summary")
            st.write(f"Total Vehicles Passed: {st.session_state['total_vehicles_passed']}")
            if st.session_state['vehicle_counts']:
                df = pd.DataFrame(list(st.session_state['vehicle_counts'].items()), columns=['Vehicle Type', 'Count'])
                chart = alt.Chart(df).mark_bar().encode(
                    x='Vehicle Type',
                    y='Count',
                    tooltip=['Vehicle Type', 'Count']
                ).properties(
                    title='Number of Vehicles Passed by Type'
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No vehicles detected in the video.")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")