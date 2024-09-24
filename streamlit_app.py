import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Initialize variables for cropping
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
resized_img5 = None
cropped_img = None

# Function to process the uploaded image and count coins
def process_image(uploaded_image):
    img = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)

    # Resize for processing if needed
    window_width, window_height = 800, 600
    aspect_ratio = img.shape[1] / img.shape[0]
    if img.shape[1] > window_width or img.shape[0] > window_height:
        if aspect_ratio > 1:
            new_width = window_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = window_height
            new_width = int(new_height * aspect_ratio)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Convert to grayscale and threshold
    input_img5_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(input_img5_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_values = {
        "50sen": {"min_radius": 34, "max_radius": 36, "value": 0.50},
        "20sen": {"min_radius": 31, "max_radius": 33, "value": 0.20},
        "10sen": {"min_radius": 28, "max_radius": 30, "value": 0.10},
        "5sen":  {"min_radius": 25, "max_radius": 27, "value": 0.05}
    }

    coin_count = {coin: 0 for coin in coin_values}
    total_value = 0.0

    # Process contours to count coins
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if 500 <= contour_area <= 5000:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)

            for coin, info in coin_values.items():
                if info["min_radius"] <= radius <= info["max_radius"]:
                    coin_count[coin] += 1
                    total_value += info["value"]
                    break

            # Draw circle around the coin
            cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), 2)

    return img, coin_count, total_value

# Streamlit App Layout
st.title("Coin Counting App")
st.write("Upload an image of coins for counting or use the webcam for real-time counting.")

# Image upload functionality
uploaded_file = st.file_uploader("Upload Coin Image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Process the uploaded image
    uploaded_image = Image.open(uploaded_file)
    processed_image, coin_counts, total_value = process_image(uploaded_image)

    # Display the processed image
    st.image(processed_image, caption='Processed Image', use_column_width=True)

    # Display coin counts and total value
    st.write("### Coin Counts")
    for coin, count in coin_counts.items():
        st.write(f"{coin}: {count}")
    st.write(f"**Total Value: RM {total_value:.2f}**")

# Real-time coin counting using webcam
if st.button("Start Real-Time Coin Counting"):
    # Set up webcam for real-time counting
    st.write("Starting webcam...")
    cap = cv2.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():
        st.error("Could not open webcam. Please check your device.")
    else:
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, coin_counts, total_value = process_image(frame_rgb)

            # Display the processed frame in Streamlit
            stframe.image(processed_frame, channels="RGB", use_column_width=True)

            # Display coin counts and total value
            st.write("### Coin Counts (Real-Time)")
            for coin, count in coin_counts.items():
                st.write(f"{coin}: {count}")
            st.write(f"**Total Value: RM {total_value:.2f}**")

            if st.button("Stop"):
                break

        cap.release()
        st.write("Webcam stopped.")
