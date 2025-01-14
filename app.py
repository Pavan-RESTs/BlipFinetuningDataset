import os
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
from io import BytesIO

if not os.path.exists("Dataset"):
    os.makedirs("Dataset")
if not os.path.exists("Dataset/Images"):
    os.makedirs("Dataset/Images")

csv_file_path = "Dataset/customDataset.csv"
if not os.path.exists(csv_file_path):
    df = pd.DataFrame(columns=["sno", "image_path", "caption"])
    df.to_csv(csv_file_path, index=False)

def handle_image_submission(image, caption):
    image_name = f"image{len(os.listdir('Dataset/Images')) + 1}.jpeg"
    image_path = os.path.join('Dataset/Images', image_name)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image.save(image_path)
    
    df = pd.read_csv(csv_file_path)
    new_entry = pd.DataFrame({
        "sno": [len(df) + 1],
        "image_path": [f"Dataset/Images/{image_name}"],
        "caption": [caption]
    })
    
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file_path, index=False)
    st.success(f"Image {image_name} saved with caption: {caption}")

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    return duration

def capture_frame_at_time(video_path, time_in_seconds):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None

st.title("Image and Video Frame Annotation for BLIP")

upload_type = st.radio("Choose upload type:", ["Image", "Video"])

if upload_type == "Image":
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        image = Image.open(image_file)
        col1, col2 = st.columns([3, 3])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state['current_image'] = image
        
        caption = st.text_input("Enter Caption for the Image")
        
        if caption and st.button("Save Image with Caption"):
            handle_image_submission(st.session_state['current_image'], caption)
            del st.session_state['current_image']

else:  
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if video_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_file.read())
        temp_file.close()

        duration = get_video_duration(temp_file.name)
        
        col1, col2 = st.columns([3, 3])
        with col1:
            st.video(video_file)
        
        selected_time = st.slider("Select time (seconds)", 0.0, duration - 1, 0.0, 0.1)
        
        frame = capture_frame_at_time(temp_file.name, selected_time)
        if frame is not None:
            st.session_state['current_frame'] = frame
            with col2:
                st.image(frame, caption=f"Frame at {selected_time:.1f} seconds", use_container_width=True)
        else:
            st.error("Failed to capture frame")
        
        caption = st.text_input("Enter Caption for the Frame")
        
        if 'current_frame' in st.session_state and caption:
            handle_image_submission(st.session_state['current_frame'], caption)
            del st.session_state['current_frame']
        
        os.unlink(temp_file.name)
