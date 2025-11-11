import streamlit as st
import pandas as pd
import plotly.express as px

# Custom sidebar title and icon
st.set_page_config(page_title="Understanding Data", page_icon="📑", layout="centered", initial_sidebar_state="expanded")

# Login protection—force user to login from main app home page
if not st.session_state.get("logged_in", False):
    st.warning("🔒 Please log in from the home page to access this page.")
    st.stop()
st.header("📑 Understanding Data")
st.write("Content for Understanding Data goes here.")

st.subheader("**Define the Problem**")
st.text("""Before starting any analysis, it’s important to understand exactly what we want to solve. This means clearly stating the question or goal and making sure it matches what the stakeholders expect. A clear problem definition helps keep the analysis focused and useful.
--identify the core problem or opportunity.
--Set clear objectives and expected outcomes.
--Understand the context, stakeholders needs and constraints.
--Define success criteria to measure the effectiveness of the analysis.""")
st.markdown("**Problem Statement**")
st.markdown(''':yellow[Before you start any analysis, it’s important to understand exactly what we want to solve.
            Gather some domain knowledge its helpful to understand the problem better.]''')
st.text_area("Paste your problem statement here:")
# paste here columns description of dataframe
st.markdown("**columns description**")
st.text_area("Paste your columns description here:", height=350)
#paste here additional about your data
st.markdown('Additionsl informational data')
st.text_area(" Paste here additional notes about your data")
st.subheader("**Common Questions About Data**")
# tabs using for common queries
tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(['Understanding Data','Understanding Graphs','Types of Graphs','Custom Measures','Advance Measures','Advance Measures in DAX'])

with tab1:
   st.markdown("""
           1. **How big is the data?**\n
              ***df.shape will give you the number of rows and columns in the data.***\n
           2. **how data looks like?**\n
              ***df.head() will give you the first 5 rows of the data.***\n
           3. **what is the data types of each columns?**\n
              ***df.dtypes will give you the data types of each columns.***\n
           4. **are there any missing values in the data?**\n
              ***df.isnull().sum() will give you the number of missing values In each columns.***\n
           5. **are there any duplicate rows in the data?**\n
              ***df.duplicated().sum() will give you the number of duplicate rows in the data.***\n
           6. **what is the statistical summary of the numeric columns?**\n
              ***df.describe() will give you the statistical summary of the numeric columns.***\n
           7. **is any correlation between columns?**\n
              ***df.corr() will give you the correlation between numeric columns.***\n
           8. **are there any outliers in the numeric columns?**\n
              ***use boxplot to visualize the outliers in the numeric columns.***\n
           9. **check how dirty is the data. like inaccurate, inconsistent, incomplete, irrelevant etc.**\n
              ***use df.info(), df.nunique(), df.value_counts() to check the data quality.***\n
           10. **what are the unique values in categorical column?**\n
               ***df['column_name'].unique() will give you the unique values in the categorical columns.***\n
               ***Additionaly you can use visualization techniques to understand the data better.***
           """)
with tab2:
   st.markdown(""" 
            **Graphs are the visual tools of data analysis—they help reveal patterns, trends, and relationships hidden in raw data. The type of graph you use depends on how many variables (or data features) you are analyzing.**\n
            1. ***Univariate Graphs***
               
               Univariate means analyzing one variable at a time to understand its distribution, spread, and central tendency.

               Common univariate graphs:

               Histogram: Displays frequency distribution of a numerical variable. Example: Distribution of ages of customers.

               Box Plot: Shows median, quartiles, and outliers. Example: Examining salary ranges.

               Bar Chart: Used for categorical data. Example: Count of students in each grade.

               Purpose: Identify patterns like skewness, outliers, or overall spread of a single attribute.  
            2. ***Bivariate Graphs***
               
               Bivariate analysis studies the relationship between two variables. One variable is often treated as independent (x-axis) and the other as dependent (y-axis).

               Common bivariate graphs:

               Scatter Plot: Best for finding correlations between two numerical variables (e.g., sales vs. advertising spend).

               Line Graph: Shows trends over time for two variables (e.g., temperature vs. day).

               Box Plot (Grouped): Compares the distribution of a quantitative variable across different categories.

               Heatmap: Used to visualize correlations or frequencies between two variables.

               Purpose: Detect relationships, patterns, or dependencies between variables.

            3. ***Multivariate Graphs***
               
               Multivariate analysis explores relationships among three or more variables simultaneously. It’s useful for complex datasets.

               Common multivariate graphs:

               Pair Plot (Seaborn): Grid of scatter plots showing relationships among all variable pairs.

               3D Scatter Plot: Visualizes three continuous variables (x, y, z dimensions).

               Bubble Chart: Similar to scatter plot but with bubble size representing a third variable.

               Parallel Coordinates Plot: Shows multiple numerical variables as parallel axes.

               Heatmap (Correlation Matrix): Highlights relationships among many variables.

               Purpose: Understand how multiple factors interact or influence each other.    
               """)
with tab3:
   st.markdown('Below charts are we use during analysis')  
   graphs = [
       ["Line Chart",         "px.line()",                "data_frame, x, y, color, title",                        "Plot y vs. x; customize by grouping (e.g., by species) and coloring lines"],
       ["Bar Chart",          "px.bar()",                 "data_frame, x, y, color, title",                        "Bar heights for categorical x; can facet (split by row/col) and color by column"],
       ["Scatter Plot",       "px.scatter()",             "data_frame, x, y, color, symbol, size, title",           "Show individual data points; customize by color, marker shape, and size"],
       ["Histogram",          "px.histogram()",           "data_frame, x, color, nbins, histnorm",                  "Frequency bar plot; can control bins, normalize to percent, use overlay/group mode"],
       ["Pie Chart",          "px.pie()",                 "data_frame, names, values, color, hole",                 "Sector size for proportions; can create donut charts with hole, set opacity, customize color sequence"],
       ["Box Plot",           "px.box()",                 "data_frame, x, y, color, facet_row, facet_col, boxmode, notched", "Statistical summary (median, quartiles, outliers); can group, facet, create notched boxes"],
       ["Violin Plot",        "px.violin()",              "data_frame, x, y, color, facet_row, facet_col, box",     "Density + summary; can show inner boxplot, facet by row/col, color by group"],
       ["3D Scatter",         "px.scatter_3d()",          "data_frame, x, y, z, color, size, symbol, title",        "Points in 3D; can vary marker size, color by another dimension"],
       ["Interactive Elements", "go.Figure().update_layout(updatemenus=...)", "Buttons, dropdowns, sliders",         "Add interactivity to switch chart types, filter data views"],
   ]
   
   df_graphs = pd.DataFrame(graphs, columns=["Chart Type", "Function", "Key Parameters", "Description/Customization"])
   
   st.table(df_graphs.set_index('Chart Type'))
with tab4:
    st.info("""
    **💡 New Measure Syntax Guide**

    **1. Add a flag or scoring column:**
    - NPA Flag: `np.where(Num_Missed_Payments > 4, "NPA", "Performer")`
    - Numeric flag: `(Num_Missed_Payments > 4).astype(int)`
    - High Sale: `np.where(sales > 1000, "High", "Low")`

    **2. Math with existing columns:**
    - Profit: `sales - discount`
    - Revenue per unit: `sales / quantity`

    **3. Normalize / Standardize:**
    - Z-score: `(sales - sales.mean()) / sales.std()`

    **4. String operations:**
    - Concatenate: `category + "_" + region`
    - Convert to uppercase: `region.str.upper()`
    - Split: `region.str.split("-", expand=True)[0]`  # First part before '-'

    **5. Data type conversion:**
    - Object (string) to int: `field.astype(int)`
    - Int to float: `field.astype(float)`

    **6. Handling missing data:**
    - Mark missing: `np.where(pd.isna(field), "Missing", field)`
    - Fill with zero: `field.fillna(0)`

    **7. Binning/Grouping (e.g., age bands):**
    - Age bins with labels:
    - pd.cut(age, bins=, labels=['child', 'teen', 'young', 'adult'])
        
    **8. Boolean mask with multiple conditions:**
    - Flag: `np.where((sales > 500) & (region == "East"), "Target", "Other")`
    - Average order value: `orders.groupby('region')['sales'].transform('mean')`
    - Proportion of high-value orders: `(orders['sales'] > 1000).mean()`
    ++9. 

    **Tips:**
    - Use `np.where` for if-else logic, `pd.cut` for grouping/bins, `.astype()` for type conversion.
    - Use column names exactly as shown in your dataset.
    
    """)
with tab5:
    st.info("""
    **💡 Advanced Custom Measure Examples**

    **Date and Time:**
    - Day of week: `date_column.dt.day_name()`
    - Month: `date_column.dt.month`
    - Days since event: `(pd.Timestamp('today') - date_column).dt.days`

    **Ranking and Percentiles:**
    - Rank by sales: `sales.rank()`
    - Percentile of sales: `sales.rank(pct=True)`

    **Rolling and Cumulative Calculations:**
    - Moving average (window=3): `sales.rolling(window=3).mean()`
    - Cumulative sum: `sales.cumsum()`

    **Count and Existence Checks:**
    - Count keyword in comment: `comment.str.contains("refund").astype(int)`
    - Count missing: `field.isna().sum()`
    - Unique values: `field.nunique()`

    **Categorical Mapping and Encoding:**
    - Map region name to code:
    region.map({'North': 1, 'South': 2, 'East': 3, 'West': 4})
    
    text
    - One-hot encode simple category flag:
    np.where(region == "North", 1, 0)
    
    text
    
    **Feature Combinations:**
    - Interaction variable: `sales * discount`
    - Difference between columns: `max_temp - min_temp`
    
    **Multi-condition classification (nested):**
    - Loan Status:  
    np.where(loan_amount == 0, "Closed",
    np.where(num_missed_payments > 4, "NPA",
    np.where(num_missed_payments > 0, "Late", "Performing")))
    
    text
    
    **Outlier Labeling:**
    - Outlier by IQR:  
    np.where((sales < sales.quantile(0.25) - 1.5sales.std()) | (sales > sales.quantile(0.75) + 1.5sales.std()), "Outlier", "Normal")
    
    text
    
    **Text Analytics:**
    - Length of review: `review.str.len()`
    - Number of words: `review.str.split().str.len()`
    
    **Math Functions:**
    - Log transform: `np.log(sales + 1)`
    - Absolute value: `score.abs()`
    - Square root: `np.sqrt(distance)`
    
    **String Formatting:**
    - Format numbers: `sales.map(lambda x: f"₹{x:,.0f}")`
    - Title case: `name.str.title()`
    
    **Aggregates with groupby (for advanced aggregation):**
    - Group mean, max, categories:  
    df.groupby("region")["sales"].mean()
    df.groupby("region")["Num_Missed_Payments"].max()
    
    text
    
    ---
    **Try combining these and create powerful new columns for dashboards, filters, or modeling!**
    """)
with tab6:
    st.info("""
    **💡 Power BI Style Measures (in pandas/numpy syntax)**

    **1. DAX-style calculations:**
    - Running Total (Cumulative Sales): `sales.cumsum()`
    - Year to Date Total: `sales[date_column.dt.year == pd.Timestamp('today').year].cumsum()`
    - Previous Value (Shift): `sales.shift(1)`
    - % Contribution: `sales / sales.sum() * 100`

    **2. Complex IF/ELSE and Nested Conditions:**
    - Multi-band score:
    np.where(sales > 1000, "High",
    np.where(sales > 500, "Medium",
    np.where(sales > 0, "Low", "Zero")))

    text
    - Churn prediction:
    np.where((last_purchase_date < pd.Timestamp('today') - pd.Timedelta(days=180)), "Churn", "Active")

    text

    **3. Date/Time Intelligence:**
    - Month Name: `date_column.dt.month_name()`
    - Fiscal Year: `date_column.dt.year + (date_column.dt.month >= 4).astype(int)`
    - Days Open: `(pd.Timestamp('today') - open_date).dt.days`

    **4. Conditional Aggregation:**
    - Total of active sales: `np.where(status == "Active", sales, 0).sum()`
    - Average for certain region: `np.where(region == "East", sales, np.nan).mean()`

    **5. Unique counts and filters:**
    - Unique customers: `customer_id.nunique()`
    - Customers with at least one return:
    df.groupby("customer_id")["return_flag"].max().sum()

    text

    **6. Dynamic buckets (age, scores):**
    - Age Group:
    pd.cut(age, bins=[0,18,35,60,np.inf], labels=["Child","Young","Adult","Senior"])

    text
    - Score band:
    pd.cut(score, bins=, labels=["F","D","C","B","A"])

    text

    **7. Ratio and percent calculations:**
    - Discount percent: `discount / sales * 100`
    - Profit margin: `(sales - cost) / sales * 100`

    **8. Top N Flag:**
    - Top 10 sales:
    np.where(sales.rank(ascending=False) <= 10, "Top10", "Other")

    text

    **9. Rolling and Lagged aggregates:**
    - 7-day rolling mean: `sales.rolling(window=7).mean()`
    - Compare to previous period: `sales / sales.shift(1)`

    **10. Text logic for categorization:**
    - Flag contains keyword: `np.where(comment.str.contains("urgent", case=False), "Urgent", "Normal")`
    - Count keywords in text: `comment.str.count("refund")`

    ---
    **Power BI and pandas/numpy both support multi-step calculated columns—try chaining logic for robust feature engineering and reporting!**
    """)


