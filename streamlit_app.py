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
    st.title("Circular ROI Detection and Pricing")

    # Use camera input to capture an image
    image_file = st.camera_input("Capture Image")

    if image_file is not None:
        # Read and process the uploaded image
        input_image = Image.open(image_file)
        input_image = np.array(input_image)

        # Process the image
        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
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
                    circular_roi = cv2.bitwise_and(input_image, input_image, mask=mask)
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped_circular_roi = circular_roi[y:y+h, x:x+w]

                    price, price_display = calculate_price(radius)
                    total_price += price
                    object_count += 1

        # Display results
        st.image(input_image, caption="Captured Image", use_column_width=True)
        st.write(f"Total number of coins found: {object_count}")
        st.write(f"Total price: RM {total_price:.2f}")

if __name__ == "__main__":
    main()
