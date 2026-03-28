import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load YOLO model (fast and lightweight)
model = YOLO("yolov8n.pt")

st.title("AI Object Detection System")

option = st.sidebar.selectbox(
    "Select Detection Mode",
    ["Image Detection", "Video Detection", "Webcam Detection"]
)

# ---------------- IMAGE DETECTION ----------------

if option == "Image Detection":

    uploaded_file = st.file_uploader("Upload Image")

    if uploaded_file:

        image = Image.open(uploaded_file)

        results = model(image)

        detected_image = results[0].plot()

        st.image(detected_image, caption="Detected Image")


# ---------------- VIDEO DETECTION ----------------

elif option == "Video Detection":

    uploaded_video = st.file_uploader("Upload Video")

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        frame_count = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Skip frames for faster processing
            if frame_count % 5 != 0:
                continue

            # Resize frame for faster inference
            frame = cv2.resize(frame, (640, 360))

            results = model(frame)

            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()


# ---------------- WEBCAM DETECTION ----------------

elif option == "Webcam Detection":

    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        results = model(frame)

        frame = results[0].plot()

        stframe.image(frame, channels="BGR")

    cap.release()