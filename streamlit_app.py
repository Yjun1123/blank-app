import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
import os

# Initialize global variables
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
resized_img5 = None
cropped_img = None

# Set up Streamlit layout
st.title("Coin Counting Application")

# Function for real-time coin counting using webcam
def realtime_coin_counting():
    st.write("Real-time Coin Counting Mode")
    cap = cv2.VideoCapture(0)  # Start webcam feed

    min_area = 4500
    max_area = 10000

    coin_values = {
        "50sen": {"min_radius": 50, "max_radius": 56, "value": 0.50},
        "20sen": {"min_radius": 45, "max_radius": 49, "value": 0.20},
        "10sen": {"min_radius": 42, "max_radius": 44, "value": 0.10},
        "5sen":  {"min_radius": 39, "max_radius": 42, "value": 0.05}
    }

    frame_placeholder = st.empty()  # Placeholder for video frame

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Webcam not detected!")
            break

        frame = np.maximum(frame, 10)
        foreground = frame.copy()
        seed = (10, 10)

        # Use floodFill to remove the background
        cv2.floodFill(foreground, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
        gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        cntrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        coin_count = {coin: 0 for coin in coin_values}
        total_value = 0.0
        total_coins_detected = 0

        circular_contours = []
        for cnt in cntrs:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))

            if 0.7 < circularity <= 1.2 and min_area <= area <= max_area:
                circular_contours.append(cnt)
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                radius = int(radius)
                center = (int(x), int(y))

                for coin, info in coin_values.items():
                    if info["min_radius"] <= radius <= info["max_radius"]:
                        coin_count[coin] += 1
                        total_value += info["value"]
                        total_coins_detected += 1
                        break

                # Draw circle around the coin
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

        # Convert frame to BGR to RGB for displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Display results
        st.write(f"Total Coins Detected: {total_coins_detected}")
        st.write(f"Total Value: RM {total_value:.2f}")

        if st.button('Stop'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Mouse cropping function (streamlined for streamlit)
def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, cropped_img

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            x_end, y_end = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False

        cropped_img = resized_img5[y_start:y_end, x_start:x_end]

        # Show the cropped image
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        cv2.imwrite('cropped_image.jpg', cropped_img)
        st.success("Cropped image saved as 'cropped_image.jpg'")
        cv2.destroyAllWindows()

# Upload image coin counting function
def upload_image_coin_counting():
    global resized_img5

    uploaded_file = st.file_uploader("Choose an image of coins", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Create a temporary file to store the image
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        input_img5 = cv2.imread(tfile.name)
        window_width, window_height = 800, 600
        aspect_ratio = input_img5.shape[1] / input_img5.shape[0]

        if input_img5.shape[1] > window_width or input_img5.shape[0] > window_height:
            if aspect_ratio > 1:
                new_width = window_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = window_height
                new_width = int(new_height * aspect_ratio)
            resized_img5 = cv2.resize(input_img5, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized_img5 = input_img5.copy()

        # Display the uploaded image in Streamlit
        st.image(cv2.cvtColor(resized_img5, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Coins"):
            detect_coins(resized_img5)

def detect_coins(image):
    st.write("Detecting coins in the image...")

    # Convert to grayscale
    input_img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(input_img_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    input_img_color = cv2.cvtColor(input_img_gray, cv2.COLOR_GRAY2BGR)
    object_count = 0
    total_value = 0
    count_50sen, count_20sen, count_10sen, count_5sen = 0, 0, 0, 0

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if 500 <= contour_area <= 5000:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if 34 <= radius <= 36:
                coin_value, coin_label = 0.50, "50 sen"
                count_50sen += 1
            elif 31 <= radius <= 33:
                coin_value, coin_label = 0.20, "20 sen"
                count_20sen += 1
            elif 28 <= radius <= 30:
                coin_value, coin_label = 0.10, "10 sen"
                count_10sen += 1
            elif 25 <= radius <= 27:
                coin_value, coin_label = 0.05, "5 sen"
                count_5sen += 1
            else:
                coin_value, coin_label = 0, "Unknown"

            total_value += coin_value
            cv2.circle(input_img_color, center, radius, (0, 255, 0), 2)
            cv2.putText(input_img_color, coin_label, (center[0] - 20, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            object_count += 1

    st.write(f"Total number of coins found: {object_count}")
    st.write(f"Total value of the coins: RM {total_value:.2f}")
    st.write(f"50 sen coins: {count_50sen}, 20 sen coins: {count_20sen}, 10 sen coins: {count_10sen}, 5 sen coins: {count_5sen}")

    # Display the image with coin classification
    st.image(cv2.cvtColor(input_img_color, cv2.COLOR_BGR2RGB), caption="Classified Coins", use_column_width=True)

# Main part of the app
option = st.sidebar.selectbox(
    "Choose mode", ("Realtime Coin Counting", "Upload Image Coin Counting")
)

if option == "Realtime Coin Counting":
    realtime_coin_counting()
elif option == "Upload Image Coin Counting":
    upload_image_coin_counting()
