import streamlit as st
import cv2
from ultralytics import YOLO

# 1. Config and Title
st.set_page_config(page_title="ARQM Live Demo", page_icon="📹")
st.title("ARQM: Real-Time Quality Check")
st.write("Turn on the webcam to see live AI detection.")

# 2. Load Model
@st.cache_resource
def load_model():
    # Replace model with your model file path
    return YOLO('QualityCheck.pt') 
    # return YOLO('best.pt')
    # return YOLO('ProductRecognition.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model not found! Check your file path.")

# 3. UI Controls
run_camera = st.checkbox('Start Camera')

frame_window = st.image([])

# 4. The Loop
if run_camera:
    camera = cv2.VideoCapture(0) # 0 is usually the default webcam
    
    if not camera.isOpened():
        st.error("Could not open webcam. Check if camera is connected or already in use.")
    else:
        while run_camera:
            # Read a frame from the webcam
            success, frame = camera.read()
            
            if not success:
                st.warning("Could not read from webcam.")
                break

            # Run YOLO on the raw frame
            results = model(frame)

            # Draw the boxes (results[0].plot() returns the image with boxes drawn)
            annotated_frame = results[0].plot()

            # Convert colors (OpenCV uses BGR, Streamlit needs RGB)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Update the placeholder with the new image
            frame_window.image(annotated_frame_rgb)
    
    # Cleanup when user unchecks the box
    camera.release()