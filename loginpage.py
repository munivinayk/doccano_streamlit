import streamlit as st
import pandas as pd
# Security related libraries
#passlib,hashlib,bcrypt,scrypt
import hashlib
import socket
import base64

#Key input variables
icon = './images/Proliflux.ico'
proliflux_logo = './images/Proliflux.png'
login_image = './images/login.jpg'
host_name = socket.gethostname()
IP_address = socket.gethostbyname(host_name)
login_threshold = 30000
banner = './images/vbanner.jpg'
menu_index = 0

#To encode text to hashes (password credentials to hashes)
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

#To validate credentials 
def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()

#postgres DB
def connect_postgres(host,port,dbname,password,user):
    import psycopg2
    conn_pg = psycopg2.connect(
        host='localhost',
        port=54320,
        dbname='mlbench_credentials',
        password='Sairamtpt@007',
        user='postgres',
    )
    conn_pg.autocommit = True
    cur = conn.cursor()

#timestamp
from datetime import datetime

def time_stamp():
    timestamp = datetime.now()
    #timestamp = datetime.timestamp(now)
    return timestamp

# DB  Functions
def create_usertable():
	#c.execute('DROP TABLE userstable')
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT, firstname TEXT, lastname TEXT, emailid TEXT, country TEXT)')
    
def create_credentialstable():
    #c.execute('DROP TABLE credentialstable')
    c.execute('CREATE TABLE IF NOT EXISTS credentialstable(username TEXT, hostname TEXT, IP_address TEXT, timestamp INTEGER)')

def login_info(username,host_name,IP_address,timestamp):
	c.execute('INSERT INTO credentialstable(username,hostname,IP_address,timestamp) VALUES (?,?,?,?)',(username,host_name,IP_address,timestamp))
	conn.commit()

def add_userdata(username,password,firstname,lastname,emailid,country):
	c.execute('INSERT INTO userstable(username,password,firstname,lastname,emailid,country) VALUES (?,?,?,?,?,?)',(username,password,firstname,lastname,emailid,country))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def view_login_info():
    c.execute('SELECT * FROM credentialstable ORDER by timestamp DESC')
    logininfo = c.fetchall()
    return logininfo

#to filter pandas table
def filter_pandas(tablename,fieldname,criteria,head):
    is_filter = tablename[fieldname]==criteria
    fiter_data= clean_login_details[is_filter]
    if head >= 0:
        filter_data_head = filter_data.head[head]
        return filter_data_head
    else:
        return filter_data

#Banner info for the home page
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    height: 100px;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

st.set_page_config(page_title='ML-Developer Workbench', page_icon = icon, initial_sidebar_state = 'auto')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

#Logo on the Side Bar
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
from PIL import Image
image = Image.open(proliflux_logo)
st.sidebar.image(image,width=250)
st.markdown('<style>h1{color: 	#000080;}</style>', unsafe_allow_html=True)
#st.markdown('<i class="material-icons">face</i>', unsafe_allow_html=True)
st.sidebar.title("ML Developer Workbench")
image1 = Image.open(login_image)
st.sidebar.image(image1, width=150, align='center')
st.sidebar.markdown(

"""
Login and Sign-up page:
Use this to create a user account and login to the ML developer work bench.
"""
)

#to supress streamlit warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

def main():
	"""ML Developer workbench Login Page"""
    
    #HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

	#title and icon information
    

	menu = ["Home","Login","SignUp"]
	choice = st.sidebar.selectbox("",menu,index=menu_index)
	#st.image(banner,width=500, height=100)
    
    Host = "LAZ-SG-L-M-8129.local"
    port = 8502
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(b'Hello, world')
        data = s.recv(1024)
    
    st.write(repr(data))

	if choice == "Home":
		st.subheader("User login info")
		login_details = view_login_info()
		clean_login_details = pd.DataFrame(login_details,columns=["Username","Host Name", "IP Address","Timestamp"])
		#st.dataframe(clean_login_details)
		#latest_hn = filter_pandas("clean_login_details","Host Name",host_name,1)
		is_hn = clean_login_details['Host Name']==host_name
		hn_filter= clean_login_details[is_hn]
		latest_hn = hn_filter.head(1)
		current_user = latest_hn.Username[0]
		latest_ts = latest_hn.Timestamp[0]
		timestamp_recent = datetime.strptime(latest_ts,'%Y-%m-%d %H:%M:%S.%f')
		timestamp_now = time_stamp()
		ts_diff = datetime.timestamp(timestamp_now) - datetime.timestamp(timestamp_recent)
		#ts_diff = timestamp_recent - timestamp_now
		#st.write(ts_diff)
        
		if ts_diff < login_threshold:
			st.success("Welcome {}".format(current_user)+" ...!!!!")
			st.subheader("Current login info")
			st.dataframe(latest_hn)
        
		else:
			st.warning ("Proceed to login section in the sidebar to access Home Page")
			#menu_index = 1
        
        #st.write()
        #now = datetime.now
        

	elif choice == "Login":
		st.subheader("ML Developer Workbench - Login")

		username = st.text_input("User Name")
		password = st.text_input("Password",type='password')
		if st.button("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:
				st.success("Logged in as {}".format(username))
				st.balloons()
				create_credentialstable()
				timestamp = time_stamp()
				login_info(username,host_name,IP_address,timestamp)
				#menu_index = 0
			else:
				st.warning("Incorrect Username/Password. Please select SignUP from the sidebar if you are new user")


	elif choice == "SignUp":
		st.subheader("ML Developer Workbench - Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')
		first_name = st.text_input("First Name")
		last_name = st.text_input("Last Name")
		email_id= st.text_input("email-id")
		country = st.text_input("Country")

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password),first_name,last_name,email_id,country)
			st.success("You have successfully created your Account")
			st.info("Go to Login Menu to login")


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

            
            
if __name__ == '__main__':
	main()