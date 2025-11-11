import streamlit as st
import random
import smtplib
import ssl
from email.message import EmailMessage

# ====== CONFIGURE YOUR EMAIL & APP PASSWORD HERE ======
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_ADDRESS = "krishnachandhan12@gmail.com"       # Your Gmail address
EMAIL_PASSWORD = "rhioryoeybjpzheh"                 # Valid Gmail App Password

def send_otp(to_email, otp):
    msg = EmailMessage()
    msg.set_content(f"Here Your login OTP code is: {otp}\nThank you for using our app.")
    msg['Subject'] = "Your One-Time Login/Registration OTP"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

def generate_otp():
    return str(random.randint(100000, 999999))

def login_flow():
    st.subheader("Login Account")
    st.text_input("Enter your email for login", key="login_email")
    if st.button("Send OTP (login)", key="login_send_otp_btn"):
        email = st.session_state.login_email
        if email:
            otp = generate_otp()
            st.session_state.login_otp = otp
            send_otp(email, otp)
            st.success("OTP sent to your email.")
        else:
            st.error("Please enter your email!")

    if "login_otp" in st.session_state:
        st.text_input("Enter OTP from your email", key="login_otp_input")
        if st.button("Verify & Log in", key="login_verify_btn"):
            user_otp = st.session_state.login_otp_input
            if user_otp == st.session_state.login_otp:
                st.session_state.logged_in = True
                st.session_state.logged_email = st.session_state.login_email
                st.success("Login successful! Redirecting...")
                #st.experimental_rerun()  # rerun to switch to default/dashboard
            else:
                st.error("Incorrect OTP. Try again.")

def logout_flow():
    st.subheader("Logout")
    if st.button("Log out", key="logout_btn"):
        st.session_state.logged_in = False
        st.success("Logged out successfully!")
        st.experimental_rerun()

def main_app():
    st.header("🏠 Home/Dashboard")
    st.success(f"You are logged in as **{st.session_state.get('logged_email','')}**")
    # Add your dashboard or default page code here
    logout_flow()

# ====== MAIN STREAMLIT LOGIC ======
st.title("Login with OTP")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()    # Show dashboard/home after login
else:
    login_flow()  # Show login page by default
