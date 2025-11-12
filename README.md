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
   - Open the displayed local URL in your browser (https://data-world.streamlit.app)

## Security Note

- Credentials (email, passwords) should NEVER be directly in your code or pushed to GitHub.
- Place all sensitive information inside your `.env` file.
- The `.gitignore` file will prevent accidental uploads of `.env`.
- Read the comments in the code for details on loading environment variables.

---

Feel free to add or modify features in `README.md` as 

## Screenshots

### 1. Home Page / Upload Dataset
![Home Page]()
*Upload your CSV file to begin data analysis*

### 2. Data Preview
![Data Preview]()
*View your uploaded data in an interactive table*

### 3. Summary Statistics
![Summary Statistics]()
*Get comprehensive statistical analysis of your dataset*

### 4. Data Visualizations
![Data Visualizations]()
*Create interactive charts: line, bar, scatter, pie, and more*

### 5. Data Cleaning
![Data Cleaning]()
*Handle missing values and clean your data*

### 6. Export Results
![Export Results]()
*Download your processed data for further use*

---

**To add your screenshots:** Drag and drop your image files into the editor above, and GitHub will automatically insert the image URLs in the `()` parentheses.your app grows!
