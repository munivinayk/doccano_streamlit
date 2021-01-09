"""
Pre-requisites
# Install the latest release of Haystack in your own environment 
#! pip install farm-haystack

# Install the latest master of Haystack
!pip install git+https://github.com/deepset-ai/haystack.git
!pip install urllib3==1.25.4

# Recommended: Start Elasticsearch using Docker
#! docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2


"""

#haystack related
from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever

import streamlit as st
import json
import requests
from elasticsearch import Elasticsearch
es = Elasticsearch()
import os

#Key input variables
icon = './images/Proliflux.ico'
proliflux_logo = './images/Proliflux.png'
login_image = './images/login.jpg'
extract_pic = './images/search.png'
#host_name = socket.gethostname()
#IP_address = socket.gethostbyname(host_name)
login_threshold = 30000
banner = './images/vbanner.jpg'
doc_dir = "./downloads"
main_dir = "."
menu_index = 0
elsatic_url = "http://localhost:9200/"

index_url = es.cat.indices(h='index', s='index').split()

def st_file_selector(st_placeholder, path='.', label='Please, select a file/folder...'):
    # get base path (directory)
    base_path = '.' if path is None or path is '' else path
    base_path = base_path if os.path.isdir(
        base_path) else os.path.dirname(base_path)
    base_path = '.' if base_path is None or base_path is '' else base_path
    # list files in base path directory
    files = os.listdir(base_path)
    if base_path is not '.':
        files.insert(0, '..')
    files.insert(0, '.')
    selected_file = st_placeholder.selectbox(
        label=label, options=files, key=base_path)
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    if selected_file is '.':
        return selected_path
    if os.path.isdir(selected_path):
        selected_path = st_file_selector(st_placeholder=st_placeholder,
                                         path=selected_path, label=label)
    return selected_path

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


#title and icon information
st.set_page_config(page_title='ML Search Engine', page_icon = icon, initial_sidebar_state = 'auto')
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

#To supress Streamlit warnings
st.set_option('deprecation.showfileUploaderEncoding', False)


st.sidebar.image(image,width=250)
st.markdown('<style>h1{color: 	#000080;}</style>', unsafe_allow_html=True)
#st.markdown('<i class="material-icons">face</i>', unsafe_allow_html=True)
st.sidebar.title("ML Developer Workbench")
image1 = Image.open(extract_pic)
st.sidebar.image(image1,width=150, align='center')
st.sidebar.markdown(

"""
Search Engine:
Use this app to index and search the content from the documents 
"""
)

menu = ["Index","Search"]
choice = st.sidebar.selectbox("",menu,index=menu_index)

if choice == "Index":
    st.subheader("ML Search Engine - Indexing Module")
    st_placeholder = st.empty()
    file_path = st_file_selector(st_placeholder, path=main_dir, label='Please, select a file/folder...')
    st.write("you have selected the following folder to index the documents: " + file_path)
    index_selection = st.sidebar.radio("Please select the Index option",("New Index", "Use Existing Index"))
    
    if index_selection == "New Index":
        elastic_index = st.sidebar.text_input("Index")
    
    elif index_selection == "Use Existing Index":
        elastic_index = st.sidebar.selectbox("Index", index_url)
    
    else:
        st.write ("you did not select the index option from the side bar")
    
    index_docs = st.button("Index Documents")
    if index_docs:
        try:
            BL_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index = elastic_index)
            dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
            BL_store.write_documents(dicts)
            st.success("Documents successfully indexed to the Index: " + elastic_index)
        except:
            st.warning("Please select a valid index from the sidebar")
        

elif choice == "Search":
    st.subheader("ML Search Engine - Search Module")
    elastic_index = st.sidebar.selectbox("Index", index_url)
    
    #input parameters for the search
    question = st.text_area("Please type your question","")
    search = st.button("Search")
    top_results = st.sidebar.slider("Top Results",1,10)
    top_reader = st.sidebar.slider("Top Reader",1,10)
    
    if search:
        st.spinner('searching the documents...')
        BL_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index = elastic_index)
        retriever = ElasticsearchRetriever(document_store=BL_store)
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
        finder = Finder(reader, retriever)
        prediction = finder.get_answers(question=question, top_k_retriever=top_results, top_k_reader= top_reader)
        st.write(prediction, details="minimal")
        st.balloons()

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
	