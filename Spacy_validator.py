import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import io
import os
import shutil
from os import listdir
from os.path import isfile, join
#import PyPDF2
import base64
#from pdfminer.high_level import extract_pages
#from pdfminer.layout import LTTextContainer, LTChar
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
import base64
import numpy as np
#from st_annotated_text import annotated_text
#import spacy_streamlit
#from spacy_streamlit import visualize_ner

#Key input variables
icon = './images/Proliflux.ico'
proliflux_logo = './images/Proliflux.png'
login_image = './images/login.jpg'
extract_pic = './images/validate.png'
#host_name = socket.gethostname()
#IP_address = socket.gethostbyname(host_name)
login_threshold = 30000
banner = './images/vbanner.jpg'
model_dir = './Spacy models'


SPACY_MODEL_NAMES = [os.path.join(model_dir, o) for o in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir,o))]

#SPACY_MODEL_NAMES = ["BL", "BL_v2", "BL_v3", "BL_v4"]
DEFAULT_TEXT = "Please Enter the extracted PDF text"

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)

#title and icon information
st.set_page_config(page_title='ML-Developer Workbench', page_icon = icon, initial_sidebar_state = 'auto')

#to download the data frame
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">download to csv</a>'
    return href

from PIL import Image
image = Image.open(proliflux_logo)
st.sidebar.image(image,width=250)
st.markdown('<style>h1{color: 	#000080;}</style>', unsafe_allow_html=True)
#st.markdown('<i class="material-icons">face</i>', unsafe_allow_html=True)
st.sidebar.title("ML Developer Workbench")
image1 = Image.open(extract_pic)
#st.sidebar.image(image1, width=150, align='center')
st.sidebar.markdown(
    """
Step4 - Spacy Model Validation:
Use this app for the Fourth stage of MLOpps life cycle i.e. validating the trained Spacy Model using docanno json file in the previous step.
"""
)

#main window elements
st.subheader("ML Opps developer work - validation")
spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()

text = st.text_area("Text to analyze", DEFAULT_TEXT, height=120)
doc = process_text(spacy_model, text)

if "parser" in nlp.pipe_names:
    st.header("Dependency Parse & Part-of-speech tags")
    st.sidebar.header("Dependency Parse")
    split_sents = st.sidebar.checkbox("Split sentences", value=True)
    collapse_punct = st.sidebar.checkbox("Collapse punctuation", value=True)
    collapse_phrases = st.sidebar.checkbox("Collapse phrases")
    compact = st.sidebar.checkbox("Compact mode")
    options = {
        "collapse_punct": collapse_punct,
        "collapse_phrases": collapse_phrases,
        "compact": compact,
    }
    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
    for sent in docs:
        html = displacy.render(sent, options=options)
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        if split_sents and len(docs) > 1:
            st.markdown(f"> {sent.text}")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

if "ner" in nlp.pipe_names:
    st.header("Named Entities")
    st.sidebar.header("Named Entities")
    default_labels = nlp.get_pipe("ner").labels
    labels = st.sidebar.multiselect(
        "Entity labels", nlp.get_pipe("ner").labels, default_labels
    )
    html = displacy.render(doc, style="ent", options={"ents": labels})
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
    if "entity_linker" in nlp.pipe_names:
        attrs.append("kb_id_")
    data = [
        [str(getattr(ent, attr)) for attr in attrs]
        for ent in doc.ents
        if ent.label_ in labels
    ]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)
    st.markdown(get_table_download_link(df), unsafe_allow_html=True)
    


if "textcat" in nlp.pipe_names:
    st.header("Text Classification")
    st.markdown(f"> {text}")
    df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
    st.dataframe(df)


vector_size = nlp.meta.get("vectors", {}).get("width", 0)
if vector_size:
    st.header("Vectors & Similarity")
    st.code(nlp.meta["vectors"])
    text1 = st.text_input("Text or word 1", "apple")
    text2 = st.text_input("Text or word 2", "orange")
    doc1 = process_text(spacy_model, text1)
    doc2 = process_text(spacy_model, text2)
    similarity = doc1.similarity(doc2)
    if similarity > 0.5:
        st.success(similarity)
    else:
        st.error(similarity)

st.header("Token attributes")

if st.button("Show token attributes"):
    attrs = [
        "idx",
        "text",
        "lemma_",
        "pos_",
        "tag_",
        "dep_",
        "head",
        "ent_type_",
        "ent_iob_",
        "shape_",
        "is_alpha",
        "is_ascii",
        "is_digit",
        "is_punct",
        "like_num",
    ]
    data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)


st.header("JSON Doc")
if st.button("Show JSON Doc"):
    st.json(doc.to_json())

st.header("JSON model meta")
if st.button("Show JSON model meta"):
    st.json(nlp.meta)