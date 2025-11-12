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
    ```sh
    git clone https://github.com/krishnachandan2412/Data-World.git
    cd Data-World
    ```

2. **Install requirements:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up your `.env` file:**
   - Create a file named `.env` in the project folder.
   - Add your email credentials:
     ```
     EMAIL_ADDRESS=your_email@example.com
     EMAIL_PASSWORD=your_password
     ```
   - Ensure `.env` is **NOT** uploaded to GitHub (it is included in `.gitignore`).

4. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

5. **Access your app:**
   - Open the displayed local URL in your browser or visit https://data-world.streamlit.app.

## Security Note

- Credentials (email, passwords) should NEVER be directly included in your code or pushed to GitHub.
- Place all sensitive information inside your `.env` file.
- The `.gitignore` file prevents accidental uploads of `.env`.
- Read the comments in the code for details on loading environment variables.

---

Feel free to add or modify features in `README.md` as your app grows!

## Screenshots

### 1. Login Page

<img width="1763" height="823" alt="Screenshot 2025-11-12 194433" src="https://github.com/user-attachments/assets/6d54c003-7edc-4eff-ab48-3d11dd94590d" />

- Login with your registered email.

### 2. Menu Bar

<img width="310" height="832" alt="Screenshot 2025-11-12 194637" src="https://github.com/user-attachments/assets/06ff3769-0019-4070-89ef-0a279b150ae5" />

- Here you can access the basic features of this application.

### 3. Upload Dataset from Local Machine

<img width="1774" height="767" alt="Screenshot 2025-11-12 194701" src="https://github.com/user-attachments/assets/695d67cd-ba9d-4b2c-acc6-20d7e75c8841" />

- It accepts all types of files, including CSV, PDF, JPG, PNG, DOC, EXCEL, JSON, etc.
- Upload your file to begin data analysis.

### 4. Upload Dataset from MySQL Server

<img width="1673" height="794" alt="Screenshot 2025-11-12 194735" src="https://github.com/user-attachments/assets/98191f54-23cc-4689-b893-2ad360c2f121" />

- It works similarly to standard SQL queries when loading files for analysis.
- *Supports only remote SQL servers (Does not support localhost).*

### 5. Data Inspection

<img width="1626" height="783" alt="Screenshot 2025-11-12 194927" src="https://github.com/user-attachments/assets/73471407-f7c6-44f4-9eb5-931fe6b3f913" />

- Preview of the dataset.

<img width="1061" height="473" alt="Screenshot 2025-11-12 195010" src="https://github.com/user-attachments/assets/e4fc0ad7-efad-406f-951e-4eb50b64b19f" />

- Overview of the dataset.
- Shows basic information about the dataset, including:
  - Outliers, missing values, duplicates, memory usage, rows and columns, etc.

<img width="1002" height="635" alt="Screenshot 2025-11-12 195203" src="https://github.com/user-attachments/assets/c638cac2-9451-4402-aea7-f89b1b3b6b77" />

- Missing values.

<img width="1006" height="591" alt="Screenshot 2025-11-12 195239" src="https://github.com/user-attachments/assets/e2fd8afd-c993-4251-acfb-b1c4d561612b" />

<img width="971" height="594" alt="Screenshot 2025-11-12 195332" src="https://github.com/user-attachments/assets/7dc2aa44-be0a-4bf9-a386-317bd74885ea" />

<img width="1034" height="621" alt="Screenshot 2025-11-12 195350" src="https://github.com/user-attachments/assets/1db1666c-6051-403f-aeae-4e137d8a9f4a" />

- Outlier detection.

<img width="962" height="580" alt="Screenshot 2025-11-12 195412" src="https://github.com/user-attachments/assets/f22e16a5-e904-4462-a2c0-831f7ce6141f" />

- Summary.

<img width="941" height="743" alt="Screenshot 2025-11-12 195447" src="https://github.com/user-attachments/assets/90b895e2-65d4-4666-b9a4-3e4ddf23cca5" />

- Top and bottom rows.

<img width="917" height="507" alt="Screenshot 2025-11-12 195502" src="https://github.com/user-attachments/assets/32c536c1-a93f-4235-9c89-c9aa8623aedf" />

- Data types of each column.

<img width="906" height="525" alt="Screenshot 2025-11-12 195517" src="https://github.com/user-attachments/assets/91f47200-1a2f-4c7c-a031-56fb45f8efde" />

- Missing values in each column.

<img width="929" height="514" alt="Screenshot 2025-11-12 195532" src="https://github.com/user-attachments/assets/766e2b1c-65ab-476d-86ac-cda14383c832" />

- Unique values in each column.

<img width="950" height="673" alt="Screenshot 2025-11-12 195545" src="https://github.com/user-attachments/assets/d4b570f4-5433-4913-86e1-921a7a8ab340" />

- Correlation between columns.

<img width="943" height="750" alt="Screenshot 2025-11-12 195946" src="https://github.com/user-attachments/assets/bb5514f2-dfcb-4557-b4b2-13698b48048d" />

<img width="1055" height="581" alt="Screenshot 2025-11-12 200006" src="https://github.com/user-attachments/assets/0fe95a2b-e78b-4111-ad37-1c9efa5a5add" />

- Count of unique values visualized as charts.

### 6. Data Preprocessing

<img width="1627" height="777" alt="Screenshot 2025-11-12 200056" src="https://github.com/user-attachments/assets/5accf80b-3429-4173-bc04-952744076771" />

<img width="951" height="575" alt="Screenshot 2025-11-12 200113" src="https://github.com/user-attachments/assets/8fa47748-bff0-4e83-a2cf-73586099c7c4" />

- Handling missing values, including:
  - Mean, mode, median, backfilling, forward filling, custom values, simple and KNN imputation, etc.

<img width="993" height="755" alt="Screenshot 2025-11-12 200149" src="https://github.com/user-attachments/assets/c6f91bd5-67db-4401-a3a2-66d4bba2764b" />

<img width="1027" height="788" alt="Screenshot 2025-11-12 200314" src="https://github.com/user-attachments/assets/6738155b-0651-407d-a5b9-46b125717a66" />

- Handling outliers.

<img width="995" height="769" alt="Screenshot 2025-11-12 200438" src="https://github.com/user-attachments/assets/c48ef8b6-e2dd-4135-9c7b-bd5a06555aa6" />

- Dealing with duplicates.

<img width="1074" height="638" alt="Screenshot 2025-11-12 200529" src="https://github.com/user-attachments/assets/837bb763-f3fa-4c42-8015-1a0e96f6eb9e" />

<img width="978" height="572" alt="Screenshot 2025-11-12 200603" src="https://github.com/user-attachments/assets/324e4b04-ae9e-40ea-b779-5febee031921" />

<img width="967" height="480" alt="Screenshot 2025-11-12 200624" src="https://github.com/user-attachments/assets/ed3ae022-b5c4-4e22-8c2a-671b8680f741" />

<img width="997" height="517" alt="Screenshot 2025-11-12 200639" src="https://github.com/user-attachments/assets/e52d2f69-c8eb-493f-ada5-ae82b8c78fa2" />

- Standardization & Normalization techniques, including:
  - Standard scaling, MinMax scaling, MeanAbs scaling.

<img width="1064" height="428" alt="Screenshot 2025-11-12 200711" src="https://github.com/user-attachments/assets/c71f4471-b3a6-4acd-85b7-e46b7f724eda" />

- Encoding techniques, including:
  - One-hot encoding, label encoding, ordinal encoding, and binning.

### 7. Feature Engineering

<img width="1620" height="813" alt="Screenshot 2025-11-12 200803" src="https://github.com/user-attachments/assets/de63567c-b756-4c43-87b8-2271433273d6" />

<img width="1025" height="562" alt="Screenshot 2025-11-12 200823" src="https://github.com/user-attachments/assets/4045f5ad-82b5-4d7c-a324-69338bf7b304" />

- Create new features.

<img width="1035" height="598" alt="Screenshot 2025-11-12 201117" src="https://github.com/user-attachments/assets/2c10d670-d55a-43d7-877f-f8581517aa5d" />

<img width="1122" height="535" alt="Screenshot 2025-11-12 201158" src="https://github.com/user-attachments/assets/7596d0d5-fd0d-4733-8bb7-c35bf9226427" />

<img width="947" height="402" alt="Screenshot 2025-11-12 201224" src="https://github.com/user-attachments/assets/7f429a78-9602-42de-8b43-b259744f9961" />

- Performing groupby operations on features.
- Create interactive charts—line, bar, scatter, pie, and more—using groupby columns.

### 8. Visualization

<img width="947" height="402" alt="Screenshot 2025-11-12 201224" src="https://github.com/user-attachments/assets/afb10899-f107-4d71-a464-210eb8128ed9" />

- Create interactive charts: line, bar, scatter, pie, and more, based on groupby columns.

---

**To add your screenshots:** Drag and drop your image files into the editor above, and GitHub will automatically insert the image URLs in the parentheses.
