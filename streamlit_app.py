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
        img = frame.to_ndarray(format="bgr")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   


        self.object_count = 0
        self.total_price = 0.0

        min_area = 10000
        max_area = 300000

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if min_area <= contour_area <= max_area:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                if len(approx) == 1:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    radius = int(radius)

                    if radius > 0:
                        price, price_display = calculate_price(radius)
                        self.total_price += price
                        self.object_count += 1

                        cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), 2)
                        cv2.putText(img, price_display, (int(x) - radius, int(y) - radius), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(img, f"Total coins: {self.object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Total price: RM {self.total_price:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

def main():
    st.title("Real-Time Circular ROI Detection and Pricing with WebRTC")

    if "camera_started" not in st.session_state:
        st.session_state.camera_started = False

    if st.button("Start Camera"):
        st.session_state.camera_started = True

    if st.session_state.camera_started:
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
