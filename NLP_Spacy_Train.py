import streamlit as st
#import pdfplumber
#import textract
import os
import shutil
from os import listdir
from os.path import isfile, join
#import PyPDF2
import base64
import io
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
from joblib import load
from doccano_api_client import DoccanoClient
import requests
import webbrowser
from datetime import datetime
import json
import sys
import spacy
import random
import time
import json


#Key input variables
icon = './images/Proliflux.ico'
proliflux_logo = './images/Proliflux.png'
login_image = './images/login.jpg'
extract_pic = './images/train.png'
#host_name = socket.gethostname()
#IP_address = socket.gethostbyname(host_name)
login_threshold = 30000
banner = './images/vbanner.jpg'

#values to initialise
DEFAULT_TEXT = "Please upload the json file"
DEFAULT_TEXT1 = "Please upload the json file"
DEFAULT_TEXT2 = "Please upload the json file"
DEFAULT_TEXT4 = ""
DEFAULT_TEXT5 = "./Spacy models"
TRAIN_DATA = []
FILE_TYPES = ["json","json1"]
error="False"
timestr = time.strftime("%Y%m%d-%H%M%S")
file_name = "for e.g. 'Spacy-Json'"
#Docker API for tesseract details
TESSERACT_API_IP = os.getenv("TESSERACT_API_IP", "localhost")
TESSERACT_API_PORT = os.getenv("TESSERACT_API_PORT", 8010)

#Training Parameters
model_loss = 0.1
model_batch_size = 100


API_URL = f"http://{TESSERACT_API_IP}:{TESSERACT_API_PORT}/process/"
DEMO_IMAGE = "text1.jpg"
PDF_Image_path = "PDF_Image"
pdf_image_path = ""
pdf_path = ""
TEXT_PDF2image = ""
spacy_output = ".\Spacy models"

#html paddin details
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

st.set_option('deprecation.showfileUploaderEncoding', False)
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

#sidebar related elements
from PIL import Image
image = Image.open(proliflux_logo)
st.sidebar.image(image,width=250)
st.markdown('<style>h1{color: 	#000080;}</style>', unsafe_allow_html=True)
st.sidebar.title("ML Developer Workbench")
image1 = Image.open(extract_pic)
st.sidebar.image(image1,width=250, align='center')
st.sidebar.markdown(
"""
Step3 - Training:
Use this app for the second stage of MLOpps life cycle i.e. named entity recognition training based on the labelled Spacy json file or Doccano json file.
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

st.sidebar.subheader("Spacy Training Parameters")
drop = st.sidebar.number_input("Model Loss", model_loss)
model_batch_size = st.sidebar.number_input("Batch Size", model_batch_size)

#To supress any streamlit version errors
st.set_option('deprecation.showfileUploaderEncoding', False)

#to save the file
def writetofile(text,file_name):
	with open(os.path.join('Spacy_json',file_name),'w') as f:
		f.write(text)
	return file_name

#to make the file downloadable
def make_downloadable(file_name):
	readfile = open(os.path.join("Spacy_json",file_name)).read()
	b64 = base64.b64encode(readfile.encode()).decode()
	href = 'Download File File (right-click and save as <some_name>.txt)'.format(b64)
	return href

#main window elements
st.subheader("ML Opps developer work - Training")
uploaded_file = st.file_uploader('Choose your Docanno annotated .json* file', type=FILE_TYPES)
if uploaded_file is not None:
    DEFAULT_TEXT2 = ""
    DEFAULT_TEXT2=uploaded_file.getvalue()
    
#extract and clean json text
text = st.text_area("Extracted json", DEFAULT_TEXT2, height=120)
lines = DEFAULT_TEXT2.splitlines()
text3=""

#convert the Docanno to Spacey json format line by line.
def convert_spacyformat(lines):
    text3=""
    text4=""
    train_data=[]
    for line in lines:
        x=json.loads(line)
        try:
            text=x["text"]
            lables=x["labels"]
            entities = []
            for start,end,label in lables:
                append1 = (start,end,label)
                entities.append(append1)
                each_record = (text,{"entities":entities})
            train_data.append(each_record)
        except json.decoder.JSONDecodeError:
            error=="True"
            continue          
    return train_data

if DEFAULT_TEXT2 == "Please upload the json file":
    TRAIN_DATA = ""
    
else:
    TRAIN_DATA = convert_spacyformat(lines)
    
#doccano to Spacy
if error=="False":
	text = st.text_area("Docanno to spacyformat", TRAIN_DATA, height=120)
	#st.write(error)
else:
	text = st.text_area("Docanno to spacyformat", DEFAULT_TEXT2, height=120)
	st.write("errors while converting to SpaCyformat: " + error)

#push button to save the SpaCy json file
yourdocument = st.text_input("please enter the file name (with out the extension)",file_name)
file_name = yourdocument + " -"+timestr + '.txt'
button = st.button("save the spacy json")
if button:
	#st.write(result)
	file_to_download = writetofile(text,file_name)
	st.info("Saved Result As :: {}".format(file_name))
	d_link = make_downloadable(file_to_download)
	st.markdown(d_link,unsafe_allow_html=True)
	st.success("file saved successfully")

def train_spacy(data,iterations, drop):
    #drop = int(drop)
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            st.write("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=drop,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            #st.write(itn)
            st.write(losses)
    return nlp

def model_notes(model_loss, batch_size, modelfile):
    final_notes = {"model_loss":model_loss,"batch_size":batch_size,"model_name":modelfile}
    final_notes_json = json.dumps(final_notes)
    notes_file = "model_notes.txt"
    notes_full_path=os.path.join(modelfile,notes_file)
    st.write(notes_full_path)
    f=open(notes_full_path,"w+")
    f.write(final_notes_json)


# Push button to send text to doccano
#text3 = text3.splitlines()
#st.write(text3)
#TRAIN_DATA==text3
modelfile = st.text_input("Enter your model name", DEFAULT_TEXT5)
button = st.button("Train named entity recognition SpaCy model")
if button:
    prdnlp = train_spacy(TRAIN_DATA, model_batch_size,drop)
    try:
        os.mkdir(modelfile)
    except OSError:
        st.write("creation of the directory at: "+modelfile+" failed")
    else:
        prdnlp.to_disk(modelfile)
        # Save our trained Model
        model_notes(drop, model_batch_size, modelfile)

	#Test your text
	#test_text = input("Enter your testing text: ")
	#doc = prdnlp(test_text)
	#for ent in doc.ents:
    		#print(ent.text, ent.start_char, ent.end_char, ent.label_)

        
#changes the footer text
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
	    footer:after {
	    content:'© Copyright Proliflux 2020'; 
	    visibility: visible;
	    display: block;
	    position: relative;
	    #background-color: red;
	    padding: 5px;
	    top: 2px;
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


