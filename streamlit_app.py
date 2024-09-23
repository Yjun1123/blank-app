import streamlit as st
import cv2
import numpy as np
from PIL import Image

def calculate_price(radius):
    if 160 <= radius <= 170:
        return 0.20, "20 sen"
    elif 171 <= radius <= 180:
        return 0.50, "50 sen"
    elif 150 <= radius <= 159:
        return 0.10, "10 sen"
    elif 130 <= radius <= 139:
        return 0.05, "5 sen"
    else:
        return 0.0, "No matching price"

def main():
    st.title("Real-Time Circular ROI Detection and Pricing")

    # Start webcam video
    run = st.checkbox("Run Video")
    
    # Initialize webcam
    if run:
        # Capture video from the webcam
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.write("Failed to capture video")
                break
            
            # Process the frame
            input_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            input_sharp_img = cv2.GaussianBlur(input_gray, (5, 5), 0)
            _, binary_image = cv2.threshold(input_sharp_img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            object_count = 0
            total_price = 0.0
            
            min_area = 10000
            max_area = 300000

            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if min_area <= contour_area <= max_area:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    radius = int(radius)

                    if radius > 0:
                        mask = np.zeros_like(input_gray)
                        cv2.circle(mask, (int(x), int(y)), radius, 255, thickness=-1)
                        circular_roi = cv2.bitwise_and(frame, frame, mask=mask)
                        x, y, w, h = cv2.boundingRect(contour)
                        cropped_circular_roi = circular_roi[y:y+h, x:x+w]

                        price, price_display = calculate_price(radius)
                        total_price += price
                        object_count += 1

            # Display results
            st.image(frame, channels="BGR", caption="Webcam Feed", use_column_width=True)
            st.write(f"Total number of coins found: {object_count}")
            st.write(f"Total price: RM {total_price:.2f}")

            # Stop the video when the user wants to
            if st.button("Stop Video"):
                break

        video_capture.release()
    else:
        st.write("Check the box to start the video")

if __name__ == "__main__":
    main()
