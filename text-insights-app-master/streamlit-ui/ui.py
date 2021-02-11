import streamlit as st
from PIL import Image
import io
import os
import numpy as np
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

TESSERACT_API_IP = os.getenv("TESSERACT_API_IP", "localhost")
TESSERACT_API_PORT = os.getenv("TESSERACT_API_PORT", 8000)

API_URL = f"http://{TESSERACT_API_IP}:{TESSERACT_API_PORT}/process/"
DEMO_IMAGE = "text1.jpg"


def pil_image_to_byte_array(image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, "PNG")
    return imgByteArr.getvalue()


@st.cache
def process_image(image_bytes):
    m = MultipartEncoder(
        fields={"file": ("filename", image_bytes, "image/jpeg")}
    )

    response = requests.post(
        API_URL, data=m, headers={"Content-Type": m.content_type}, timeout=8000,
    )
    return response


st.title("OCR with Tesseract")
img_file_buffer = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"]
)


if img_file_buffer is not None:
    image_bytes = pil_image_to_byte_array(Image.open(img_file_buffer))
    image_array = np.array(Image.open(img_file_buffer))

else:
    image_bytes = open(DEMO_IMAGE, "rb")
    image_array = np.array(Image.open(DEMO_IMAGE))

response = process_image(image_bytes)
if response.status_code == 200:
    processed_text = response.json()["text"]
else:
    processed_text = f"API response code {response.status_code}"

st.image(
    image_array, use_column_width=True,
)

st.header("Extracted text")
st.write(processed_text)
