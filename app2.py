import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from ultralytics import YOLO

# --- 1. Load Both Models ---
# Replace these filenames with your actual model names
# --- FUNCTION: Resize with Padding (Letterbox) ---
def resize_with_padding(img, expected_size=640):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # 2. Resize maintaining aspect ratio
    img.thumbnail((expected_size, expected_size))
    
    # 3. Calculate padding
    delta_width = expected_size - img.size[0]
    delta_height = expected_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    
    # 4. Add White padding
    return ImageOps.expand(img, padding, fill="white")
try:
    detector = YOLO('ProductRecognition.pt') 
    quality_model = YOLO('QualityCheck.pt') 
except Exception as e:
    st.error(f"Error loading models: {e}")

st.title("Automated Remote Quality Monitoring (ARQM)")
try:
    uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert file to image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        img_640 = resize_with_padding(img_array,640)
        # Show original
        st.image(image, caption="Original Image", use_container_width=True)

        # --- TIER 1: DETECTION ---
        st.write("Scanning for product presence...")
        det_results = detector(img_640)
        
        # Check if any objects were found
        if len(det_results[0].boxes) == 0:
            st.warning("❌ No product detected. Quality check skipped.")
        
        else:
            st.success("✅ Product detected! Proceeding to Quality Check...")
            
            # Visualize the detection (optional, just to show user what was found)
            # res_plotted = det_results[0].plot()
            # st.image(res_plotted, caption="Tier 1 Detections", width=300)

            # --- TIER 2: QUALITY CHECK ---
            st.write("Analyzing Installation Quality on Full Image...")
            
            # Pass the WHOLE image (img_array) to the second model
            qual_results = quality_model(img_640)
            result = qual_results[0]
            # Check if any defect/quality objects were found
            if len(result.boxes) > 0:
                # 1. Get the class ID of the most confident detection
                box = result.boxes[0]           # Get the first object found
                class_id = int(box.cls[0])      # Get its ID number (e.g., 0, 1, 2)
                class_name = result.names[class_id] # Look up the name (e.g., "Bent", "Scratch")
                confidence = float(box.conf[0]) # Get confidence (e.g., 0.85)

                # 2. Display logic based on the name
                # If your model detects defects (like "Bent"), then finding something is BAD.
                st.error(f"Installation Quality FAILED: Found '{class_name}' ({confidence:.0%})")
                
                # Show the image with the red box around the defect
                res_plotted_bgr = result.plot()
                res_plotted_rgb = res_plotted_bgr[:, :, ::-1] # Fix color
                st.image(res_plotted_rgb, caption="Defect Location", use_container_width=True)
                
                # Fallback if your quality model is actually an Object Detector
            else:
                # If NO boxes were found, that usually means the product is clean/good
                st.balloons()
                st.success("Installation Quality PASSED: No defects detected.")
        
        
except Exception as e:
    st.error(f"An error occurred: {e}")