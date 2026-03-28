import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load YOLO model
model = YOLO("yolov8n.pt")

st.title("YOLOv8 Object Detection")

option = st.sidebar.selectbox(
    "Select Detection Mode",
    ["Image Detection", "Video Detection", "Webcam Detection"]
)

# ---------------- IMAGE DETECTION ----------------

if option == "Image Detection":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        results = model(image)

        detected_image = results[0].plot()

        st.image(detected_image, caption="Detected Image")


# ---------------- VIDEO DETECTION ----------------

elif option == "Video Detection":

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

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

            frame = cv2.resize(frame, (640, 360))

            results = model(frame)

            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()


# ---------------- WEBCAM DETECTION ----------------

elif option == "Webcam Detection":

    st.write("Turn on your webcam for live detection")

    class VideoTransformer(VideoTransformerBase):

        def transform(self, frame):

            img = frame.to_ndarray(format="bgr24")

            img = cv2.resize(img, (640, 360))

            results = model(img)

            img = results[0].plot()

            return img

    webrtc_streamer(
        key="yolo-webcam",
        video_transformer_factory=VideoTransformer
    )