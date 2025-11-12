import streamlit as st
import pandas as pd
import mysql.connector

# page config
st.set_page_config(page_title="Data Source :cloud:", layout="centered",page_icon="☁️", initial_sidebar_state="expanded")
st.header("☁️:gray[Connect to MySQL Server]",divider=True)
st.subheader("Here you can connect to MySQL Server and set it as session data",divider=True)

# Initialize session state for form inputs if they don't exist
if 'db_config' not in st.session_state:
    st.session_state.db_config = {
        'host': 'localhost',
        'database': 'your_database',
        'username': '',
        'password': ''
    }

# Database connection inputs with session state
col1, col2 = st.columns(2)
with col1:
    host = st.text_input("MySQL Host",
                         value=st.session_state.db_config['host'],
                         key='host_input')
    database = st.text_input("Database Name",
                             value=st.session_state.db_config['database'],
                             key='db_input')
with col2:
    username = st.text_input("Username",
                             value=st.session_state.db_config['username'],
                             key='user_input')
    password = st.text_input("Password",
                             type="password",
                             value=st.session_state.db_config['password'],
                             key='pass_input')

# Update session state when inputs change
if 'host_input' in st.session_state:
    st.session_state.db_config.update({
        'host': st.session_state.host_input,
        'database': st.session_state.db_input,
        'username': st.session_state.user_input,
        'password': st.session_state.pass_input
    })

# Load saved query if it exists
if 'saved_query' not in st.session_state:
    st.session_state.saved_query = "SELECT * FROM table_name LIMIT 10;"

query = st.text_area("Enter SQL Query",
                     value=st.session_state.saved_query,
                     key='query_input')

if st.button("Run Query", type="primary"):
    if not all([st.session_state.db_config['host'],
                st.session_state.db_config['database'],
                st.session_state.db_config['username'],
                st.session_state.db_config['password']]):
        st.warning("Please fill in all database connection details and a query.")
    else:
        try:
            # Save the query
            st.session_state.saved_query = st.session_state.query_input

            # Database connection and query execution
            conn = mysql.connector.connect(
                host=st.session_state.db_config['host'],
                user=st.session_state.db_config['username'],
                password=st.session_state.db_config['password'],
                database=st.session_state.db_config['database']
            )

            cursor = conn.cursor(dictionary=True)
            cursor.execute(st.session_state.saved_query)

            # Get results
            result = cursor.fetchall()

            if result:
                # Convert to DataFrame
                df = pd.DataFrame(result)
                st.session_state['data'] = df
                st.session_state['data_loaded'] = True
                st.session_state['data_source'] = 'mysql'

                st.success(" Query executed successfully! Data loaded into session state.")
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("Query executed successfully but no results were returned.")

            cursor.close()
            conn.close()

        except mysql.connector.Error as err:
            st.error(f"Database error: {err}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Add a debug section to check session state
if st.toggle("Show Session State (Debug)"):
    st.write("Current session state keys:", list(st.session_state.keys()))
    if 'data' in st.session_state:
        st.write("DataFrame shape:", st.session_state['data'].shape)
        st.write("DataFrame columns:", st.session_state['data'].columns.tolist())
    st.write("Current DB Config:", st.session_state.get('db_config', 'Not set'))
    st.write("Saved Query:", st.session_state.get('saved_query', 'No query saved'))