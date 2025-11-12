# Data-World

A Data Analytics Streamlit Web App for interactive data exploration and visualization.

## Features

- Upload CSV datasets for instant analysis.
- Summary statistics and data profiling.
- Data cleaning and missing value handling options.
- Flexible data visualizations: line, bar, scatter, pie, and more.
- Customizable filters and views for your uploaded data.
- Export processed data for further use.
- Email integration (for sending results or notifications) using secure environment variables.

## How to Use

1. **Clone the repository:**
    ```
    git clone https://github.com/krishnachandan2412/Data-World.git
    cd Data-World
    ```

2. **Install requirements:**
    ```
    pip install -r requirements.txt
    ```

3. **Set up your `.env` file:**
   - Create a file named `.env` in the project folder.
   - Add your email credentials:
     ```
     EMAIL_ADDRESS=your_email@example.com
     EMAIL_PASSWORD=your_password
     ```
   - Ensure `.env` is **NOT** uploaded to GitHub (it's in `.gitignore`).

4. **Run the Streamlit app:**
    ```
    streamlit run app.py
    ```

5. **Access your app:**
   - Open the displayed local URL in your browser (usually http://localhost:8501)

## Security Note

- Credentials (email, passwords) should NEVER be directly in your code or pushed to GitHub.
- Place all sensitive information inside your `.env` file.
- The `.gitignore` file will prevent accidental uploads of `.env`.
- Read the comments in the code for details on loading environment variables.

---

Feel free to add or modify features in `README.md` as your app grows!
