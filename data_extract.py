"""
Pre-requisites
Need to set-up the docker tesseract engine

Got to the tesseract-engine folder and run the following commands:

docker build -t tesseract-engine .
docker run -p 8010:8010 tesseract-engine
From the root directory of this repo run: curl -X POST "http://127.0.0.1:8000/process/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@assets/text1.jpg;type=image/jpeg" which should return:

{
  "success": "true",
  "text": "This is the first line of\nthis text example.\n\nThis is the second line\nof the same text."
}

docker run -p 8010:8010 58ab621d11b4
"""
import time
from enum import Enum
from io import BytesIO, StringIO
from typing import Union 
import pandas as pd
import streamlit as st
import base64
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import io
import textract
import numpy as np
import os
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL import Image 
#from iolib import load
from joblib import load
from doccano_api_client import DoccanoClient
import shutil
from os import listdir
from os.path import isfile, join
import webbrowser
from datetime import datetime
import json
import time
from PIL import Image, ImageEnhance
import streamlit as st
import pandas as pd
# Security related libraries
#passlib,hashlib,bcrypt,scrypt
import hashlib
import socket
import base64
from io import StringIO
import speech_recognition as sr
import pyttsx3
import cv2
from pdf2image import convert_from_path, convert_from_bytes
import array
from pathlib import Path
import streamlit.components.v1 as stc
import docx2txt



#Key input variables
icon = './images/Proliflux.ico'
proliflux_logo = './images/Proliflux.png'
login_image = './images/login.jpg'
extract_pic = './images/extract.png'
#host_name = socket.gethostname()
#IP_address = socket.gethostbyname(host_name)
login_threshold = 30000
banner = './images/vbanner.jpg'
DEFAULT_TEXT = "Please upload a PDF document"
DEFAULT_TEXT1 = "Please upload a PDF document"
DEFAULT_TEXT2 = "Please upload a file of type: png, jpg, pdf, jpeg"
DEFAULT_TEXT3 = "Please upload a PDF document"
DEFAULT_TEXT4 = "11"
DEFAULT_TEXT5 = ""
DEFAULT_TEXT6 = ""
filename = ""
timestr = time.strftime("%Y%m%d-%H%M%S")
#project_id = "11"
file_classification = ""

#Docker API for tesseract details
TESSERACT_API_IP = os.getenv("TESSERACT_API_IP", "localhost")
TESSERACT_API_PORT = os.getenv("TESSERACT_API_PORT", 8010)

API_URL = f"http://{TESSERACT_API_IP}:{TESSERACT_API_PORT}/process/"
DEMO_IMAGE = "text1.jpg"
PDF_Image_path = "PDF_Image"
pdf_image_path = ""
pdf_path = ""
TEXT_PDF2image = ""
doccano_url = "https://share.streamlit.io/munivinayk/doccano_streamlit/main/Docanno_conn.py"


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

#To write to the text file
def writetofile(text,file_name):
	with open(os.path.join('downloads',file_name),'w') as f:
		f.write(text)
	return file_name

#To make the text file downloadable
def make_downloadable(filename):
	readfile = open(os.path.join("downloads",filename)).read()
	b64 = base64.b64encode(readfile.encode()).decode()
	href = 'Download File File (right-click and save as <some_name>.txt)'.format(b64)
	return href

#To display files in a particular path
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

#convert image to byte array
def pil_image_to_byte_array(image):
    imgByteArr = io.BytesIO()
    try:
        image.save(imgByteArr, "PNG")
        return imgByteArr.getvalue()
    except:
        return imgByteArr.getvalue()

#call the tesseract api
@st.cache
def process_image(image_bytes):
    m = MultipartEncoder(
        fields={"file": ("filename", image_bytes, "image/jpeg")}
    )

    response = requests.post(
        API_URL, data=m, headers={"Content-Type": m.content_type}, timeout=8000,
    )
    return response

#understand the format of the file
def file_classify(stringio):
    global file_classification
    stringio = stringio.strip()
    stringio = stringio.replace("    "," ")
    stringio = stringio.replace("  "," ")
    stringio = stringio.replace("  "," ")
    stringio = stringio.replace("  "," ")
    stringio = stringio.replace("  "," ")
    stringio = stringio.replace("%","")
    splitted = stringio.split()
    first = splitted[0]
    second = splitted[1]
    combine = first+" "+second
    combine_strip = first.strip()
    st.write(combine_strip)
    if "JFIF" in combine_strip:
        file_classification = file_classification+"image"
    elif "PDF" in combine_strip:
        file_classification = file_classification+"pdf"
    elif "ID3" in combine_strip:
        file_classification = file_classification+"mp3"
    elif "PNG" in combine_strip:
        file_classification = file_classification+"image"
    elif "RIFFK" in combine_strip:
        file_classification = file_classification+"wav"
    elif "M4A" in combine_strip:
        file_classification = file_classification+"m4a"
    elif "PK" in combine_strip:
        file_classification = file_classification+"word"
    elif ">" in combine_strip:
        file_classification = file_classification+"word_old"
    return file_classification

#Function to convert text to speech
def SpeakText(command): 
      
    # Initialize the engine 
    engine = pyttsx3.init() 
    engine.say(command)  
    engine.runAndWait()
    
#function related to pre-processing of images

#To enhance the image
@st.cache
def enhance_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)
    #blur = cv2.medianBlur(img,5)
    #erode = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,11,2)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(img,(3,3))
    _,thresh = cv2.threshold(blur,240,255,cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
    erode = cv2.erode(thresh,element,3)
    cv2.destroyAllWindows()
    return thresh

@st.cache
def enhance_OCR_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    line_items_coordinates = []
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)

        if y >= 600 and x <= 1000:
            if area > 10000:
                image = cv2.rectangle(img, (x,y), (2200, y+h), color=(255,0,255), thickness=3)
                line_items_coordinates.append([(x,y), (2200, y+h)])

        if y >= 2400 and x<= 2000:
            image = cv2.rectangle(img, (x,y), (2200, y+h), color=(255,0,255), thickness=3)
            line_items_coordinates.append([(x,y), (2200, y+h)])
    cv2.destroyAllWindows()
    return image

#To upscale the image - 300 dpi
@st.cache
def set_image_dpi(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    im = cv2.imdecode(file_bytes, 0)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    cv2.destroyAllWindows()
    return im_resized

#To denoise the image
@st.cache
def remove_noise_and_smooth(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    cv2.destroyAllWindows()
    return or_image

#PDF to Image to extract content missed from direct extract
    

#title and icon information
st.set_page_config(page_title='ML-Developer Workbench', page_icon = icon, initial_sidebar_state = 'auto')
# The actual page title always gets " ∙ Streamlit" appended.
# We can discuss whether to remove this later.

#Hides the default menu and rerun icons
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

from PIL import Image
image = Image.open(proliflux_logo)


st.set_option('deprecation.showfileUploaderEncoding', False)


st.sidebar.image(image,width=250)
st.markdown('<style>h1{color: 	#000080;}</style>', unsafe_allow_html=True)
#st.markdown('<i class="material-icons">face</i>', unsafe_allow_html=True)
st.sidebar.title("ML Developer Workbench")
image1 = Image.open(extract_pic)
st.sidebar.image(image1,width=150, align='center')
st.sidebar.markdown(

"""
Step1 - Data Extraction:
Use this app for the first stage of MLOpps life cycle i.e. Data extraction from the readable PDF and upload into the Doccanno backend.
"""
)

# Push button to check Doccano connection
button = st.sidebar.button("check doccano connection")
if button:
	f=open("./doccano_cred.txt","r")
	credentials = f.readline()
	f.close()
	y = json.loads(credentials)
	ts_old = y["timestamp"]
	url = y["url"]
	id = y["id"]
	pw = y["pw"]
	now = datetime.now()
	ts_now = datetime.timestamp(now)
	diff = (ts_now-ts_old)

	if diff < 30000:
		doccano_client = DoccanoClient(
		url,
		id,
		pw
		)
		st.success("Connected to Doccano")    
		st.balloons()
		project_id = st.text_input("Please enter project ID - sequence", DEFAULT_TEXT4)
	else:
		url2 = 'http://localhost:8502/'
		webbrowser.open_new_tab(url2)



st.subheader("ML Opps developer work - Data Extraction")
format_types=["png", "jpg", "pdf", "jpeg", "mp3", "wav", ".docx"]

file = st.file_uploader("Upload file", type=format_types)
show_file = st.empty()


if not file:
    show_file.info("Please upload a file of type: " + ", ".join(format_types))
    

elif file is not None:
    DEFAULT_TEXT2 = ""
    bytes_data = file.getvalue()
    encode_text = st.text_area("Encoded_text",bytes_data)
    stringio = bytes_data.decode("utf-8", errors="ignore")
    file_classification = file_classify(stringio)
    st.write(file_classification)
    decode_text = st.text_area("Decoded_text",stringio)
    
    
    if file_classification == "":
        show_file.info("Please upload a file of type: " + ", ".join(["png", "jpg", "pdf", "jpeg"]))
    
    
    elif file_classification == ("image"):
        #sharpen = enhance_image (file)
        
        #for image enhancement and bounding boxes
        #sharpen = enhance_OCR_image(file)
        #st.image(sharpen, use_column_width=True,)
        
        #denoise_image = remove_noise_and_smooth (file)
        #st.image(denoise_image, use_column_width=True,)
        image_bytes = pil_image_to_byte_array(Image.open(file))
        image_array = np.array(Image.open(file))
        response = process_image(image_bytes)
        if response.status_code == 200:
            DEFAULT_TEXT2 = response.json()["text"]
        else:
            DEFAULT_TEXT2 = f"API response code {response.status_code}"
        st.image(image_array, use_column_width=True,)
    
    
    elif file_classification == ("pdf"):
        #global pdf_image_path
        base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
        for page_layout in extract_pages(file):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    DEFAULT_TEXT2=DEFAULT_TEXT2+element.get_text()

        pdf_image_Path = convert_from_bytes(bytes_data, dpi=300, output_folder = PDF_Image_path, single_file=True, fmt='PNG',paths_only=True)
        st.write(pdf_image_Path)
        ##pdf_path = os.path.join(PDF_Image_path,"8bc6f694-2067-4c28-b655-77ad06c73078.jpg")
        #pdf_path = Path(pdf_image_Path)
        ##st.write(pdf_path)
        ##pdf_image = open(pdf_path, "rb")
        #st.write(pdf_image)
        ##pdf_io = pdf_image.read()
        #st.write("-----------------------------------------")
        #st.write(pdf_io)
        #image_bytes = bytearray(pdf_io)
        #st.write(image_text)
        #st.image(pdf_image,use_column_width=True,)
        ##image_bytes = pil_image_to_byte_array(Image.open(pdf_io))
        
        #image_bytes = pil_image_to_byte_array(Image.open(pdf_image_Path))
        ##image_array = np.array(Image.open(file))
        ##response = process_image(image_bytes)
        ##if response.status_code == 200:
            ##Text_PDF2image = response.json()["text"]
        ##else:
            ##Text_PDF2image = f"API response code {response.status_code}"
        #st.image(image_array, use_column_width=True,)
        
        if len(TEXT_PDF2image) < len(DEFAULT_TEXT2):
            DEFAULT_TEXT2 == DEFAULT_TEXT2
        else:
            DEFFAULT_TEXT2 == TEXT_PDF2image
        
    elif file_classification == ("mp3" or "wav" or "m4a"):
        format_file = "audio/"+file_classification
        st.write(format_file)
        st.audio(bytes_data, format = format_file)
        sr.recognizer()
        audio = r.record(bytes_data)
        DEFAULT_TEXT2 = r.recognize_google(audio)

    elif file_classification == ("word"):
        try:
            DEFAULT_TEXT2 = docx2txt.process(file)
            #st.write(DEFAULT_TEXT2)
        except:
            DEFFAULT_TEXT2 = str(file.read(),"utf-8")

lines = DEFAULT_TEXT2.split("\n")
non_empty_lines = [line for line in lines if line.strip() != ""]

string_without_empty_lines = ""
for line in non_empty_lines:
    string_without_empty_lines += line.strip() + " "
    
    
DEFAULT_TEXT3=string_without_empty_lines.strip()
DEFAULT_TEXT3=DEFAULT_TEXT3.replace('"',">")
DEFAULT_TEXT3=DEFAULT_TEXT3.replace('  '," ")
DEFAULT_TEXT3=DEFAULT_TEXT3.replace('  '," ")
DEFAULT_TEXT3=DEFAULT_TEXT3.replace('  '," ")
DEFAULT_TEXT3=DEFAULT_TEXT3.replace('  '," ")

#text = st.text_area("Extracted text - bytecode", DEFAULT_TEXT, height=120)
#text = st.text_area("Extracted text - base64", DEFAULT_TEXT1, height=120)
text1 = st.text_area("Extracted text - plain text", DEFAULT_TEXT2, height=120)
text2 = st.text_area("Cleaned text", DEFAULT_TEXT3, height=120)

project_id = st.text_input("Please enter Doccano project number","")
# Push button to send text to doccano
button = st.button("upload the cleaned text to doccanno")

if button:
	f=open("./doccano_cred.txt","r")
	credentials = f.readline()
	f.close()
	y = json.loads(credentials)
	ts_old = y["timestamp"]
	url = y["url"]
	id = y["id"]
	pw = y["pw"]
	now = datetime.now()
	ts_now = datetime.timestamp(now)
	diff = (ts_now-ts_old)

	if diff < 30000:
		doccano_client = DoccanoClient(
		url,
		id,
		pw
		)
		
		#project_id = st.text_input("Please enter Doccano project ID - sequence", DEFAULT_TEXT4)
		data = {
            		'text': text2
        		}
        
		add_url = f'{url}/v1/projects/{project_id}/docs'
		add_url = add_url.replace("//","/")
		add_url = add_url.replace("https:/","https://")
		st.write(add_url)
		response = doccano_client.post(add_url, json=data)
		st.success("cleaned text successfully uploaded to doccano")    
		st.balloons()
	else:
		#url2 = 'http://localhost:8502/'
		webbrowser.open_new_tab(doccano_url)

#push button to save the cleaned text
yourdocument = st.text_input("To download file - please enter the file name (with out the extension)",filename)
file_name = yourdocument + " -"+timestr + '.txt'
button = st.button("save the cleaned text")

if button:
	#st.write(result)
	file_to_download = writetofile(text2,file_name)
	st.info("Saved Result As :: {}".format(file_name))
	d_link = make_downloadable(file_to_download)
	st.markdown(d_link,unsafe_allow_html=True)
	st.success("file saved successfully")

#changes the footer text
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
	    footer:after {
	    content:'© Copyright Proliflux 2021'; 
	    visibility: visible;
	    display: block;
	    position: relative;
	    #background-color: red;
	    padding: 5px;
	    top: 2px;
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
	
