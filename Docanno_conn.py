import streamlit as st
import os
from os import listdir
from os.path import isfile, join
import io
from doccano_api_client import DoccanoClient
import requests
import json
from datetime import datetime
import hashlib
import socket
import base64
import pandas as pd


#Key input variables
icon = './images/Proliflux.ico'
proliflux_logo = './images/Proliflux.png'
connect_pic = './images/connect.png'
#host_name = socket.gethostname()
#IP_address = socket.gethostbyname(host_name)
login_threshold = 30000
banner = './images/vbanner.jpg'

DEFAULT_TEXT4 = ""
DEFAULT_TEXT5 = ""
DEFAULT_TEXT6 = ""
ts_now = datetime.now()
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

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
st.sidebar.image(image,width=250)
st.markdown('<style>h1{color: 	#000080;}</style>', unsafe_allow_html=True)
st.sidebar.title("Connect to Doccanno")
image1 = Image.open(connect_pic)
st.sidebar.image(image1,width=150, align='center')
st.sidebar.markdown(
"""
Annotation back end:
Use this app to connect to Doccano client (backend).
"""
)

st.set_option('deprecation.showfileUploaderEncoding', False)

#button1 = st.button("Retrieve Credentials")
#if button1:
st.subheader("Doccano connection form")
f=open("doccano_cred.txt","r")
credentials = f.readline()
f.close()
y = json.loads(credentials)
ts_old = y["timestamp"]
now = datetime.now()
ts_now = datetime.timestamp(now)
diff = (ts_now-ts_old)
if diff < 30000:
	DEFAULT_TEXT4 = y["url"]
	DEFAULT_TEXT5 = y["id"]
	DEFAULT_TEXT6 = y["pw"]
else:
	DEFAULT_TEXT4 = ""
	DEFAULT_TEXT5 = ""
	DEFAULT_TEXT6 = ""


url = st.text_input("Please enter Doccano Url", DEFAULT_TEXT4)
id = st.text_input("Please enter Doccano user id", DEFAULT_TEXT5)
pw = st.text_input("Please enter Doccano pw", DEFAULT_TEXT6, type="password")

agree = st.checkbox("Remember details")

		
button2 = st.button("establish Doccano connection")
if button2:
	with st.spinner('Connecting to docanno...'):
		# instantiate a client and log in to a Doccano instance
		doccano_client = DoccanoClient(
		url,
		id,
		pw
		)

		# Success message
		st.success("Connected to Doccano")    
		st.balloons()
		if agree:
			f=open("./doccano_cred.txt","w+")
			timestamp = datetime.timestamp(now)
			x={"url":url,"id":id,"pw":pw,"status":"success","timestamp":timestamp}
			y = json.dumps(x)
			f.write(y)
			f.close()

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

		
	
