import numpy as np
import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split


# Page configuration
st.set_page_config(page_title="Data Cleaning Tool", page_icon="✨", layout="centered",initial_sidebar_state="expanded")

# Check if data is available in session state
st.header("🧹:blue[Data Cleaning Tool]",divider=True)
st.subheader("Here you can clean your data",divider=True)
if 'data' not in st.session_state or st.session_state['data'] is None:
    st.warning("No data found. Please load data first.")
    st.stop()

df = st.session_state['data']
st.success(" Session DataFrame loaded!")
st.dataframe(df.head())  # Display a sample of the dataframe

# Copy data before cleaning
if 'original_df' not in st.session_state:
    st.session_state['original_df'] = df.copy()

# Main tabs
tab1, tab2, tab3, tab4 , tab5= st.tabs([
    '🔍 Missing Values',
    '📊 Outliers',
    '🔍 Duplicates',
    '🔍 Scaling',
    '🔍 Encoding',

])


# Helper function to detect outliers using IQR
def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)


with tab1:  # Missing Values Tab
    st.subheader("Handling Missing Values",divider=True)
    from sklearn.impute import SimpleImputer
    from sklearn.impute import KNNImputer

    # Missing values analysis
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum().values,
        'Missing %': (df.isnull().mean() * 100).round(2)
    }).sort_values('Missing Values', ascending=False)

    if not missing_df[missing_df['Missing Values'] > 0].empty:
        st.dataframe(missing_df[missing_df['Missing Values'] > 0])

        # Column selection
        cols_with_missing = missing_df[missing_df['Missing Values'] > 0]['Column'].tolist()
        cols_to_process = st.multiselect(
            "Select columns to process:",
            options=cols_with_missing,
            default=cols_with_missing
        )

        if cols_to_process:
            # Replacement method
            method = st.radio(
                "Choose how to handle missing values:",
                ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Custom Value", "Remove Rows","Simple impute","KNN impute","Remove Columns"],
                horizontal=True
            )

            custom_value = None
            if method == "Custom Value":
                custom_value = st.text_input("Enter value to fill missing data:")

            if st.button("Apply Missing Value Treatment", key="fill_na"):
                try:
                    df_clean = df.copy()  # Create a clean copy to work with

                    for col in cols_to_process:
                        if method == "Mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
                            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                        elif method == "Median" and pd.api.types.is_numeric_dtype(df_clean[col]):
                            df_clean[col].fillna(df_clean[col].median(), inplace=True)
                        elif method == "Mode":
                            mode_val = df_clean[col].mode()
                            if not mode_val.empty:
                                df_clean[col].fillna(mode_val[0], inplace=True)
                        elif method == "Forward Fill":
                            df_clean[col].fillna(method='ffill', inplace=True)
                        elif method == "Backward Fill":
                            df_clean[col].fillna(method='bfill', inplace=True)
                        elif method == "Custom Value" and custom_value is not None:
                            # Try to convert to float if possible, otherwise keep as string
                            try:
                                df_clean[col].fillna(float(custom_value), inplace=True)
                            except ValueError:
                                df_clean[col].fillna(custom_value, inplace=True)

                    # Handle row removal after processing all columns
                    if method == "Remove Rows":
                        initial_rows = len(df_clean)
                        df_clean = df_clean.dropna(subset=cols_to_process)
                        rows_removed = initial_rows - len(df_clean)
                        st.session_state['data'] = df_clean
                        st.success(f" Removed {rows_removed} rows with missing values.")
                        st.rerun()
                    elif method == "Remove Columns":
                        df_clean = df_clean.drop(columns=cols_to_process)
                        st.session_state['data'] = df_clean
                        st.success(f" Removed {len(cols_to_process)} columns with missing values.")
                        st.rerun()
                    elif method == "Simple impute":
                        imputer = SimpleImputer(strategy='mean')
                        df_clean[cols_to_process] = imputer.fit_transform(df_clean[cols_to_process])
                        st.session_state['data'] = df_clean
                        st.success(" Missing values filled successfully!")
                        st.rerun()
                    elif method == "KNN impute":
                        imputer = KNNImputer(n_neighbors=5)
                        df_clean[cols_to_process] = imputer.fit_transform(df_clean[cols_to_process])
                        st.session_state['data'] = df_clean
                        st.success(" Missing values filled successfully!")
                        st.rerun()
                    else:
                        st.session_state['data'] = df_clean
                        st.success(" Missing values filled successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f" Error: {str(e)}")
    else:
        st.success(" No missing values found in the dataset!")

# Remove the last line that was causing an error:
# st.session_state['data'] = df.fix errors

with tab2:  # Outliers Tab
    st.subheader("Outlier Detection and Treatment",divider=True)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for outlier detection.")
    else:
        # Select columns for outlier analysis
        outlier_cols = st.multiselect(
            "Select columns for outlier analysis:",
            options=numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )

        if outlier_cols:
            # Show summary statistics
            st.write("### Summary Statistics")
            st.dataframe(df[outlier_cols].describe())

            # Show potential outliers
            st.write("### Potential Outliers")
            for col in outlier_cols:
                outliers = detect_outliers(df[col])
                if outliers.any():
                    st.write(f"**{col}**: {outliers.sum()} potential outliers detected")
                    fig = px.violin(df[col], title=f'Box Plot showing Outliers in {col}')
                    st.plotly_chart(fig)

            # Outlier treatment options
            st.write("### Outlier Treatment")
            treatment = st.radio(
                "Choose how to handle outliers:",
                ["None", "Remove", "Cap", "Replace with NA"],
                horizontal=True
            )

            if treatment != "None":
                if st.button("Apply Outlier Treatment"):
                    df_clean = df.copy()
                    try:
                        for col in outlier_cols:
                            if treatment == "Remove":
                                df_clean = df_clean[~detect_outliers(df_clean[col])]
                            elif treatment == "Cap":
                                Q1 = df_clean[col].quantile(0.25)
                                Q3 = df_clean[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                            elif treatment == "Replace with NA":
                                df_clean[col] = df_clean[col].mask(detect_outliers(df_clean[col]))

                        st.session_state['data'] = df_clean
                        st.success(f" Outlier treatment applied successfully! {len(df) - len(df_clean)} rows removed.")
                        st.rerun()
                    except Exception as e:
                        st.error(f" Error: {str(e)}")

with tab3:  # Duplicates Tab
    st.subheader("Duplicate Data Handling",divider=True)

    # Show duplicate information
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        st.warning(f"Found {dup_count} duplicate rows in the dataset.")
        if st.checkbox("Show duplicate rows"):
            st.dataframe(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))

        # Duplicate removal options
        st.write("### Remove Duplicates")
        keep_option = st.radio(
            "Which duplicates to keep:",
            ["First occurrence", "Last occurrence", "None (remove all duplicates)"],
            index=0
        )

        subset_cols = st.multiselect(
            "Consider only these columns for duplicate detection:",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )

        if st.button("Remove Duplicates"):
            try:
                keep = 'first' if "First" in keep_option else 'last' if "Last" in keep_option else False
                df_clean = df.drop_duplicates(keep=keep, subset=subset_cols if subset_cols else None)
                st.session_state['data'] = df_clean
                st.success(f" Removed {len(df) - len(df_clean)} duplicate rows.")
                st.rerun()
            except Exception as e:
                st.error(f" Error: {str(e)}")
    else:
        st.success(" No duplicate rows found in the dataset.")
with tab4:
    try:


        st.subheader("Standardisation & Normalization", divider=True)
        st.caption(
            "Standardisation & Normalization are techniques used to transform the data into a common scale, making it easier to compare and analyze.")
        # Instead of using the original df or previously loaded data,
        # load the scaled data from session state:


        # Initial load: use session_state data if available, else original df
        df = st.session_state.get('data', df)


        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        select_numeric_cols = st.multiselect(
            "Select numeric columns for scaling",
            options=numeric_cols,
            default=numeric_cols
        )
        if not select_numeric_cols:
            st.write("Please select at least one numeric column for scaling.")
            st.stop()

        X = df[select_numeric_cols]
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        select_scale = st.selectbox(
            "Select scaling method",
            ["Standard scaling", "Min-Max scaling", "MaxAbs scaling", "Robust scaling", "None"]
        )

        scaler_dict = {
            "Standard scaling": StandardScaler(),
            "Min-Max scaling": MinMaxScaler(),
            "MaxAbs scaling": MaxAbsScaler(),
            "Robust scaling": RobustScaler()
        }

        if select_scale == "None":
            st.write("No scaling method selected.")
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
        else:
            scaler = scaler_dict[select_scale]
            scaler.fit(X_train)
            X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            st.write("Before Scaling:")
            st.dataframe(X_train.describe())
            st.write("After Scaling:")
            st.dataframe(X_train_scaled.describe())

            # Concatenate scaled train and test sets (keep index order for later alignment)
            X_scaled = pd.concat([X_train_scaled, X_test_scaled]).sort_index()

            # Add object columns back (keep original index order)
            obj_cols = df.select_dtypes(exclude=[np.number]).columns
            numeric_idx = X_scaled.index
            # Align numeric and object columns by their index
            obj_data_aligned = df.loc[numeric_idx, obj_cols]
            final_scaled_df = pd.concat([X_scaled, obj_data_aligned], axis=1)
            # --- Preserve original column order ---
            final_scaled_df = final_scaled_df[df.columns]
            # ---------------------------------------

            st.write("Preview of Scaled DataFrame (with object columns):")
            st.dataframe(final_scaled_df.head())

            # Save to session state ONLY IF user clicks the button
            if st.button("Save Scaled Data", key="save_scaled"):
                st.session_state['scaled_data'] = final_scaled_df
                st.session_state['data'] = final_scaled_df  # <-- updates main session data only when confirmed
                st.success("Scaled data (including object columns) saved and set as active session data.")
            else:
                st.info("Data will only be saved and replace the original if you click 'Save Scaled Data'.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
with tab5:

    # Initial load: use session_state data if available, else original df
    df = st.session_state.get('data', df)

    select_encode = st.selectbox(
        "Select encoding method",
        ["One-hot encoding", "Label encoding", "Ordinal encoding", "Binning", "None"]
    )
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer

    obj_cols = df.select_dtypes(include='object').columns.tolist()

    if select_encode == "None":
        st.write("No encoding method selected.")

    elif select_encode == "Label encoding":
        label_cols = st.multiselect("Select columns for label encoding", options=obj_cols, default=[])
        if label_cols:
            df_label = df.copy()
            le = LabelEncoder()
            for col in label_cols:
                df_label[col] = le.fit_transform(df_label[col].astype(str))
            st.write("Label Encoded DataFrame:")
            st.dataframe(df_label)
            if st.button("Save Encoded Data", key="save_label"):
                st.session_state['data'] = df_label
                st.success("Label-encoded data saved for further processing!")
            st.download_button(
                label="Download Encoded Data as CSV",
                data=df_label.to_csv(index=False).encode(),
                file_name="label_encoded_data.csv",
                mime="text/csv"
            )
        # ---------------------------
        df_processed = st.session_state.get('data', df)
        st.write("This DataFrame will be used for further processing steps.")
        st.dataframe(df_processed)
        # ---------------------------
    elif select_encode == "Ordinal encoding":
        ordinal_cols = st.multiselect("Select columns for ordinal encoding", options=obj_cols, default=[])
        if ordinal_cols:
            df_ordinal = df.copy()
            oe = OrdinalEncoder()
            for col in ordinal_cols:
                df_ordinal[col] = oe.fit_transform(df_ordinal[col].values.reshape(-1, 1))
            st.write("Ordinal Encoded DataFrame:")
            st.dataframe(df_ordinal)
            if st.button("Save Encoded Data", key="save_ordinal"):
                st.session_state['data'] = df_ordinal
                st.success("Ordinal-encoded data saved for further processing!")
            st.download_button(
                label="Download Encoded Data as CSV",
                data=df_ordinal.to_csv(index=False).encode(),
                file_name="ordinal_encoded_data.csv",
                mime="text/csv"
            )
        # ---------------------------
        df_processed = st.session_state.get('data', df)
        st.write("This DataFrame will be used for further processing steps.")
        st.dataframe(df_processed)
        # ---------------------------
    elif select_encode == "Binning":
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        select_numeric = st.multiselect(
            "Select numeric columns for binning",
            options=numeric_cols,
            default=[]
        )
        if select_numeric:
            df_binned = df.copy()
            strategy_options_list = ["quantile", "uniform", "kmeans","None"]
            strategy = st.selectbox(
                "Select strategy",
                options=strategy_options_list,
                index=strategy_options_list.index("None")
            )
            n_bins = st.number_input("Number of bins", min_value=2, max_value=10, value=5)
            if strategy == "None":
                st.write('Select a strategy')
            else:
                kbin = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
                transformed = kbin.fit_transform(df_binned[select_numeric])

                # Always do ordinal encoding and add range labels
                df_binned[select_numeric] = pd.DataFrame(transformed, columns=select_numeric, index=df_binned.index)
                for i, col in enumerate(select_numeric):
                    edges = kbin.bin_edges_[i]
                    intervals = [
                        f"[{edges[j]:.2f}–{edges[j + 1]:.2f})"
                        for j in range(len(edges) - 1)
                    ]
                    df_binned[col + "_range"] = df_binned[col].apply(
                        lambda x: intervals[int(x)] if not pd.isna(x) else None
                    )
                st.write("Binned DataFrame:")
                st.dataframe(df_binned)
                if st.button("Save Binned Data", key="save_binned"):
                    st.session_state['data'] = df_binned
                    st.success("Binned data saved for further processing!")
                st.download_button(
                    label="Download Binned Data as CSV",
                    data=df_binned.to_csv(index=False).encode(),
                    file_name="binned_data.csv",
                    mime="text/csv"
                )
        # ---------------------------
        df_processed = st.session_state.get('data', df)
        st.write("This DataFrame will be used for further processing steps.")
        st.dataframe(df_processed)
        # ---------------------------
    else:
        selected_cols = st.multiselect(
            "Select columns for one-hot encoding",
            options=obj_cols,
            default=[]
        )
        if selected_cols:
            df_encoded = pd.get_dummies(df, columns=selected_cols)
            st.write("One-hot Encoded DataFrame:")
            st.dataframe(df_encoded)
            if st.button("Save Encoded Data", key="save_onehot"):
                st.session_state['data'] = df_encoded
                st.success("One-hot encoded data saved for further processing!")
            st.download_button(
                label="Download Encoded Data as CSV",
                data=df_encoded.to_csv(index=False).encode(),
                file_name="onehot_encoded_data.csv",
                mime="text/csv"
            )

        # ---------------------------
        df_processed = st.session_state.get('data', df)
        st.write("This DataFrame will be used for further processing steps.")
        st.dataframe(df_processed)
        # ---------------------------
