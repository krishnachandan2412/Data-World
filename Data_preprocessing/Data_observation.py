import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Data Inspection", page_icon="🧾", layout="centered",initial_sidebar_state="expanded")

st.header("🔍:blue[Data Inspection]",divider=True)
st.subheader("Here you can observe the dataset",divider=True)
if 'data' in st.session_state and st.session_state['data'] is not None and not st.session_state['data'].empty:
    df = st.session_state['data']
    nrows = min(len(df), 10)
    st.success(" Session DataFrame loaded!")
    st.dataframe(df.sample(nrows))
    
    st.header(':rainbow[Basic insights of data]', divider='gray')
    st.subheader("***Overview of dataset***")
    st.markdown(f"In this dataset there are _:blue[{df.shape[1]}]_ variables and _:red[{df.shape[0]}]_ observations.")
    st.caption(f"There are total _:orange[{df.select_dtypes(include=['object']).shape[1]}]_ text variables, "
        f"_:green[{df.select_dtypes(include=['number']).shape[1]}]_ numeric variables, "
        f"_:blue[{df.select_dtypes(include=['bool']).shape[1]}]_ boolean variables and " 
        f"_:green[{df.select_dtypes(include=['datetime']).shape[1]}]_ datetime variables."
        f"The dataset contains total _:red[{df.isnull().sum().sum()}]_ missing values in terms of _:blue[{(df.isnull().sum().sum()/ (df.shape[0]*df.shape[1]))*100:.1f}%]_ of the data." 
        f"There are total _:orange[{df.duplicated().sum()}]_ duplicate rows in the dataset. "
        f"The dataset occupies _:green[{df.memory_usage(deep=True).sum() / 1024 ** 2:.3f} MB]_ of memory.")

    # Calculate missing value percentage
    missing_df = pd.DataFrame({
        'Missing columns': df.columns,
        'Missing Values': df.isnull().sum().values,
        'Missing Percentage': (df.isnull().sum().values / len(df)) * 100,
    })
    missing_nonzero = missing_df[missing_df['Missing Percentage'] > 0]

    if missing_nonzero.empty:
        st.success("Great! There are no missing values in your dataset.")
    else:
        st.markdown(f"The dataset contains _:red[{df.isnull().sum().sum()}]_ missing values out of _:green[{df.shape[0]*df.shape[1]}]_ total values. _:blue[{(df.isnull().sum().sum()/ (df.shape[0]*df.shape[1]))*100:.2f}%]_ of the data is missing.")
        st.dataframe(missing_nonzero)
        # Visualize missing columns
        fig = px.bar(missing_nonzero, x='Missing columns', y='Missing Values',
                     title='Columns with Missing Values',
                     labels={'Missing columns': 'Columns', 'Missing Values': 'Count of Missing Values'},
                     text='Missing Values',
                     template='plotly_white')
        st.plotly_chart(fig)
        fig2 = px.imshow(df.isnull(), title='Missing Values Heatmap',
                         labels=dict(x="Columns", y="Rows", color="Missing"),
                         color_continuous_scale=['blue', 'white'])
        st.plotly_chart(fig2)

    # Duplicate rows
    duplicates_percentage = (df.duplicated().sum() / df.shape[0]) * 100
    duplicates_info = df[df.duplicated()]
    if duplicates_percentage > 5:
        st.markdown(f"The dataset has a significant number of duplicate rows: _:red[{df.duplicated().sum()}]_. Use df.drop_duplicates() to remove them.")
        st.caption('Displaying duplicate rows:')
        st.dataframe(duplicates_info, width=400)
    elif 1 <= duplicates_percentage <= 5:
        st.warning("The dataset has a moderate amount of duplicate rows. Consider reviewing/removing them for data integrity.")
        st.caption('Displaying duplicate rows:')
        st.dataframe(duplicates_info, width=400)
    else:
        st.success("Great! There are no duplicate rows in the dataset.")

    # Outliers
    numeric_cols = df.select_dtypes(include=['number']).columns
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = len(outliers)
    total_outliers = sum(outlier_info.values())
    outlier_df = pd.DataFrame(list(outlier_info.items()), columns=['Outlier column', 'Outlier Count'])
    outlier_df = outlier_df[outlier_df['Outlier Count'] > 0]

    if total_outliers > 0:
        st.markdown(f"The dataset contains a total of _:red[{total_outliers}]_ outliers across numeric columns.")
        st.caption("Displayed column-wise outlier counts:")
        st.dataframe(outlier_df, width=350)
        st.markdown(':rainbow[Outliers Detection in Numeric Columns]')
        st.info("Use the multiselect box below to choose numeric columns for outlier visualization.")
        selected_cols = st.multiselect('Select numeric columns to visualize outliers:', options=numeric_cols)
        if selected_cols:
            for col in selected_cols:
                st.subheader(f'Violin Plot for {col}')
                fig = px.violin(df, y=col,
                                title=f'Violin Plot showing Outliers in {col}',
                                box=True, points="all")
                st.plotly_chart(fig)
                st.subheader(f'Scatter Plot for {col}')
                fig2 = px.scatter(df, y=col, title=f'Scatter Plot showing Outliers in {col}')
                st.plotly_chart(fig2)
        else:
            st.info("Please select at least one numeric column to visualize outliers.")
    else:
        st.success("Great! There are no outliers detected in the numeric columns of the dataset.")

    # Tabs for detailed analysis
    tab1, tab2, tab3, tab5, tab6, tab7, tab8= st.tabs(['Data Summary', 'Top and Bottom rows', 'Data types', 'Missing Values', 'Unique Values', 'Correlation Heatmap', 'Value Counts'])

    with tab1:
        with st.expander("Statistical Summary of the dataset. It gives clarity about your data"):
            st.write(f'There are {df.shape[0]} rows & {df.shape[1]} columns in dataset:')
            st.dataframe(df.describe(include='all').T, height=300, use_container_width=True)

    with tab2:
        st.subheader(':gray[Top rows of the dataset:]')
        top_rows = st.slider("Select number of top rows to view:", 1, df.shape[0])
        st.dataframe(df.head(top_rows))
        st.subheader(':gray[Bottom rows of the dataset:]')
        bottom_rows = st.slider("Select number of bottom rows to view:", 1, df.shape[0])
        st.dataframe(df.tail(bottom_rows))

    with tab3:
        st.subheader(':gray[Data Types of each column:]')
        dtypes_df = pd.DataFrame({'Column Name': df.columns, 'Data Type': df.dtypes.astype(str)}).set_index('Column Name')
        st.dataframe(dtypes_df, width=200)

    with tab5:
        st.subheader(':gray[Missing Values in each column:]')
        missing_only_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
        st.dataframe(missing_only_df, width=300, height=300)

    with tab6:
        st.subheader(':gray[Unique Values in each column:]')
        unique_df = pd.DataFrame(df.nunique(), columns=['Unique Values'])
        st.dataframe(unique_df)

    with tab7:
        st.subheader(':gray[Correlation Heatmap:]')
        if df.select_dtypes(include=['number']).shape[1] < 2:
            st.write("Not enough numerical columns to compute correlation.")
        else:
            corr = df.select_dtypes(include=['number']).corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title='Correlation Heatmap')
            st.plotly_chart(fig)

    with tab8:
        st.subheader('Count of unique values in a selected column:', divider='gray')
        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox('Choose a column for insights:', options=list(df.columns))
        with col2:
            toprows = st.number_input("Number of top rows to display:", min_value=1, value=5)
        count = st.button('Show me the result')
        if count:
            result = df[column].value_counts().head(int(toprows))
            result_df = result.reset_index()
            result_df.columns = [column, 'count']

            fig = px.bar(result_df, x=column, y='count', title=f'Top {toprows} most frequent values in {column} (Bar Chart)')
            st.plotly_chart(fig)

            fig2 = px.pie(result_df, values='count', names=column, title=f'Top {toprows} most frequent values in {column} (Pie Chart)')
            st.plotly_chart(fig2)

            fig3 = px.line(result_df, x=column, y='count', title=f'Top {toprows} most frequent values in {column} (Line Chart)')
            st.plotly_chart(fig3)

    #with tab9:
    #    st.subheader(':rainbow[Groupby : Simplify your data analysis]', divider='rainbow')
    #    st.write('The groupby lets you summarize data by specific categories and groups')
    #    with st.expander('Group By your columns'):
    #        col1, col2, col3 = st.columns(3)
    #        with col1:
    #            groupby_cols = st.multiselect('Choose your column to groupby', options=list(df.columns))
    #        with col2:
    #            operation_col = st.selectbox('Choose column for operation', options=list(df.columns))
    #        with col3:
    #            operation = st.selectbox('Choose operation',
    #                                     options=['sum', 'max', 'min', 'mean', 'median', 'count'])
#
    #        if (groupby_cols):
    #            result = df.groupby(groupby_cols).agg(
    #                newcol=(operation_col, operation)
    #            ).reset_index()
#
    #            st.dataframe(result)

else:
    st.info("No DataFrame found in session. Please upload a file and set it as session data from the upload page.")

