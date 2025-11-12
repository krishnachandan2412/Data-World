import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import pdfplumber
import docx
import json
#page config
st.set_page_config(page_title="Data Source :local", layout="centered",page_icon="📁" ,initial_sidebar_state="expanded")

if not st.session_state.get('logged_in', False):
    st.warning("Please log in from the home page to access this page.")
    st.stop()
st.header("📁:gray[ File Upload]",divider=True)

st.subheader("Here you can upload files and set it as session data",divider=True)
uploaded_files = st.file_uploader(
    "📁 Upload files",
    type=["png", "jpg", "jpeg", "pdf", "xlsx", "xls", "docx", "json", "csv"],
    accept_multiple_files=True
)

@st.cache_data
def convert_to_dataframe(file, file_type):
    if file_type.startswith("image"):
        text = pytesseract.image_to_string(Image.open(file))
        return pd.DataFrame([line.split() for line in text.split('\n') if line.strip()])
    elif file_type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            lines = [line for page in pdf.pages if (text := page.extract_text()) for line in text.split('\n')]
        return pd.DataFrame([line.split() for line in lines if line.strip()])
    elif "excel" in file_type:
        return pd.read_excel(file)
    elif "csv" in file_type:
        return pd.read_csv(file)
    elif "json" in file_type:
        content = json.load(file)
        return pd.DataFrame(content) if isinstance(content, list) else pd.DataFrame([content])
    elif "wordprocessingml" in file_type:
        doc = docx.Document(file)
        lines = [para.text for para in doc.paragraphs if para.text.strip()]
        return pd.DataFrame([line.split() for line in lines])
    else:
        return None

# Process uploaded files
dfs = []
filenames = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            df = convert_to_dataframe(uploaded_file, uploaded_file.type)
            if df is not None and not df.empty:
                dfs.append(df)
                filenames.append(uploaded_file.name)
            else:
                st.warning(f" No data extracted from {uploaded_file.name}.")
        except Exception as e:
            st.error(f" Error with {uploaded_file.name}: {e}")

# Display results in tabs
if dfs:
    file_tabs = st.tabs(filenames)
    for i, df in enumerate(dfs):
        with file_tabs[i]:
            st.dataframe(df)
            # Button to set this df as session data
            if st.button(f"load {filenames[i]} ", key=f"set_session_{i}"):
                st.session_state['data'] = df

                st.success(f"{filenames[i]} set as session data!")
