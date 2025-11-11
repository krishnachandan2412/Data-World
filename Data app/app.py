import streamlit as st
import random
import smtplib
import ssl
from email.message import EmailMessage

from streamlit import sidebar

#page config
st.set_page_config(page_title="Data World", layout="centered", initial_sidebar_state="expanded")
# ====== EMAIL CONFIGURATION ======
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_ADDRESS = "krishnachandhan12@gmail.com"
EMAIL_PASSWORD = "rhioryoeybjpzheh"  # Gmail App Password


def send_otp(to_email, otp):
    msg = EmailMessage()
    msg.set_content(f"Here is your one-time OTP to login code: {otp}\nThank you for using our app. Kindly share your feedback with us.It helps us to improve our app.")
    msg['Subject'] = "Login OTP"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:# send otp to email
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)


def generate_otp():
    return str(random.randint(100000, 999999))


# Initialize session state
if "logged_in" not in st.session_state: # check if logged_in is in session state
    st.session_state.logged_in = False # set logged_in to False
if "logged_email" not in st.session_state: # check if logged_email is in session state
    st.session_state.logged_email = "" # set logged_email to empty string


def login_flow():
    st.title("🔏:rainbow[ Login to Continue]")
    st.subheader("Explore Data world ",divider=True)
    st.caption("Use only your registered Google account Or Microsoft account")
    st.text_input("Enter your email", key="login_email")
    if st.button("Send OTP"):
        if "login_email" == "@gmail.com" or "@outlook.com":
            email = st.session_state.get("login_email", "")
            if email:
                otp = generate_otp()
                st.session_state.login_otp = otp
                send_otp(email, otp)
                st.success("OTP sent to your email.")

        else:
            st.error("Please enter your Google account or Microsoft account !")

    if "login_otp" in st.session_state: # check if otp is sent
        st.text_input("Enter OTP from your email", key="login_otp_input")# user input otp
        if st.button("Verify & Log In"):# verify otp
            typed_otp = st.session_state.get("login_otp_input", "")
            correct_otp = st.session_state.get("login_otp", "") # get otp
            if typed_otp == correct_otp and typed_otp != "": # check if otp is correct
                st.session_state.logged_in = True#Store login status
                st.session_state.logged_email = st.session_state.get("login_email", "")#Store email for next login
                st.success("Login successful! All pages are now unlocked.")
                st.session_state.pop("login_otp", None)#Remove OTPs for next login
                st.session_state.pop("login_otp_input", None)#Remove OTP input for next login
                st.rerun()
            else:
                st.error("Incorrect OTP. Try again.")


def logout_flow():
    if st.sidebar.button("🚪 Log out", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.logged_email = ""
        st.sidebar.success("Logged out. Please log in again.")
        st.rerun()


# Hide Streamlit watermark and footer using custom CSS
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define all pages with proper emoji icons
Understanding_data = st.Page(
    "Data_source\\Understanding_data.py",
    title="Understanding data",
    icon="📊",
    default=True
)

Data_source_local = st.Page(
    "Data_source\\Data_source_local.py",
    title="Data from computer",
    icon="💾"
)

Data_source_server = st.Page(
    "Data_source\\Data_source_server.py",
    title="Data from Server",
    icon="☁️"
)

Data_summary = st.Page(
    "Data_preprocessing\\Data_observation.py",
    title="Data Inspection",
    icon="🧾"
)

Data_cleaning = st.Page(
    "Data_preprocessing\\Cleaning_data.py",
    title="Data Cleaning",
    icon="🧹"
)

Feature_engineering = st.Page(
    "Data_preprocessing\\Feature_engineering.py",
    title="Feature Engineering",
    icon="⚙️"
)

Data_visualization = st.Page(
    "Data_preprocessing\\Charts_dashboard.py",
    title="Data Visualization",
    icon="📈"
)

# Main app logic
if st.session_state.logged_in:
    # Show logout button in sidebar
    logout_flow()

    # Show navigation for logged-in users
    pg = st.navigation(
        {
            "Understanding Data": [Understanding_data],
            "Data Gathering": [Data_source_local, Data_source_server],
            "Data Preprocessing": [Data_summary, Data_cleaning, Feature_engineering, Data_visualization],

        }
    )
    pg.run()
else:
    # Show login page for non-logged-in users
    login_flow()