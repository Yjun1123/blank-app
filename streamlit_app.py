import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.object_count = 0
        self.total_price = 0.0

    def transform(self, frame):
        # Convert frame to a format suitable for OpenCV
        img = frame.to_ndarray(format="bgr")
        
        # Process the frame
        input_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input_sharp_img = cv2.GaussianBlur(input_gray, (5, 5), 0)
        _, binary_image = cv2.threshold(input_sharp_img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Reset counters for each frame
        self.object_count = 0
        self.total_price = 0.0
        
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
                    circular_roi = cv2.bitwise_and(img, img, mask=mask)
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped_circular_roi = circular_roi[y:y+h, x:x+w]

                    price, price_display = calculate_price(radius)
                    self.total_price += price
                    self.object_count += 1

        # Display results on the frame
        cv2.putText(img, f"Total coins: {self.object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Total price: RM {self.total_price:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

def main():
    st.title("Real-Time Circular ROI Detection and Pricing with WebRTC")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
