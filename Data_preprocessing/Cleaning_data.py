import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, LabelEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

# --- Safe session state/data load ---
if 'data' not in st.session_state:
    st.session_state['data'] = None

# Page configuration
st.set_page_config(page_title="Data Cleaning Tool", page_icon="✨", layout="centered", initial_sidebar_state="expanded")

st.header("🧹:blue[Data Cleaning Tool]", divider=True)
st.subheader("Here you can clean your data", divider=True)

# File uploader
if st.session_state['data'] is None:
    uploaded_file = st.file_uploader("Upload your CSV file to begin:", type=['csv'])
    if uploaded_file is not None:
        st.session_state['data'] = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a file to continue.")
        st.stop()

df = st.session_state['data']
st.success("Session DataFrame loaded!")
st.dataframe(df.head())

if 'original_df' not in st.session_state:
    st.session_state['original_df'] = df.copy()

def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '🔍 Missing Values',
    '📊 Outliers',
    '🔍 Duplicates',
    '🔍 Scaling',
    '🔍 Encoding',
])

with tab1:
    st.subheader("Handling Missing Values", divider=True)
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum().values,
        'Missing %': (df.isnull().mean() * 100).round(2)
    }).sort_values('Missing Values', ascending=False)

    if not missing_df[missing_df['Missing Values'] > 0].empty:
        st.dataframe(missing_df[missing_df['Missing Values'] > 0])
        cols_with_missing = missing_df[missing_df['Missing Values'] > 0]['Column'].tolist()
        cols_to_process = st.multiselect(
            "Select columns to process:",
            options=cols_with_missing,
            default=cols_with_missing
        )

        method = st.radio(
            "Choose how to handle missing values:",
            ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Custom Value", "Remove Rows", "Simple impute", "KNN impute", "Remove Columns"],
            horizontal=True
        )
        custom_value = None
        if method == "Custom Value":
            custom_value = st.text_input("Enter value to fill missing data:")

        preview_missing = st.button("Preview", key="preview_missing")
        apply_missing = st.button("Apply", key="apply_missing")
        df_preview = None

        if (preview_missing or apply_missing) and cols_to_process:
            df_clean = df.copy()
            try:
                for col in cols_to_process:
                    if col not in df_clean.columns:
                        continue
                    if method == "Mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    elif method == "Median" and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif method == "Mode":
                        mode_val = df_clean[col].mode()
                        if not mode_val.empty:
                            df_clean[col] = df_clean[col].fillna(mode_val[0])
                    elif method == "Forward Fill":
                        df_clean[col] = df_clean[col].fillna(method='ffill')
                    elif method == "Backward Fill":
                        df_clean[col] = df_clean[col].fillna(method='bfill')
                    elif method == "Custom Value" and custom_value is not None:
                        try:
                            df_clean[col] = df_clean[col].fillna(float(custom_value))
                        except ValueError:
                            df_clean[col] = df_clean[col].fillna(custom_value)

                if method == "Remove Rows":
                    df_clean = df_clean.dropna(subset=cols_to_process)
                elif method == "Remove Columns":
                    df_clean = df_clean.drop(columns=cols_to_process, errors='ignore')
                elif method == "Simple impute":
                    imputer = SimpleImputer(strategy='mean')
                    exist_cols = [col for col in cols_to_process if col in df_clean.columns]
                    if exist_cols:
                        df_clean[exist_cols] = imputer.fit_transform(df_clean[exist_cols])
                elif method == "KNN impute":
                    imputer = KNNImputer(n_neighbors=5)
                    exist_cols = [col for col in cols_to_process if col in df_clean.columns]
                    if exist_cols:
                        df_clean[exist_cols] = imputer.fit_transform(df_clean[exist_cols])

                df_preview = df_clean

                if preview_missing:
                    st.write("Preview of DataFrame after Missing Value Treatment:")
                    st.dataframe(df_preview.head())
                if apply_missing:
                    st.session_state['data'] = df_preview
                    st.success("Missing value treatment applied!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.success("No missing values found in the dataset!")

with tab2:
    st.subheader("Outlier Detection and Treatment", divider=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_cols = st.multiselect(
        "Select columns for outlier analysis:",
        options=numeric_cols,
        default=None
    )
    treatment = st.radio(
        "Choose how to handle outliers:",
        ["None", "Remove", "Cap", "Replace with NA"],
        horizontal=True
    )
    preview_outliers = st.button("Preview", key="preview_outliers")
    apply_outliers = st.button("Apply", key="apply_outliers")
    df_preview = None

    if (preview_outliers or apply_outliers) and outlier_cols:
        df_clean = df.copy()
        try:
            for col in outlier_cols:
                if col not in df_clean.columns:
                    continue
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

            df_preview = df_clean

            if preview_outliers:
                st.write("Preview of DataFrame after Outlier Treatment:")
                st.dataframe(df_preview.head())
            if apply_outliers:
                st.session_state['data'] = df_preview
                st.success("Outlier treatment applied!")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Show stats/plots only for existing & selected columns
    existing_outlier_cols = [col for col in outlier_cols if col in df.columns]
    if existing_outlier_cols:
        st.write("### Summary Statistics")
        st.dataframe(df[existing_outlier_cols].describe())
        st.write("### Potential Outliers")
        for col in existing_outlier_cols:
            outliers = detect_outliers(df[col])
            if outliers.any():
                st.write(f"**{col}**: {outliers.sum()} potential outliers detected")
                fig = px.violin(df[col], title=f'Box Plot showing Outliers in {col}')
                st.plotly_chart(fig)
    else:
        st.info("No selected columns are present in the DataFrame for outlier analysis.")

with tab3:
    st.subheader("Duplicate Data Handling", divider=True)
    dup_count = df.duplicated().sum()
    subset_cols = st.multiselect(
        "Consider only these columns for duplicate detection:",
        options=df.columns.tolist(),
        default=None
    )
    keep_option = st.radio(
        "Which duplicates to keep:",
        ["First occurrence", "Last occurrence", "None (remove all duplicates)"],
        index=0
    )

    preview_dup = st.button("Preview", key="preview_dup")
    apply_dup = st.button("Apply", key="apply_dup")
    df_preview = None

    if (preview_dup or apply_dup) and subset_cols:
        try:
            keep = 'first' if "First" in keep_option else 'last' if "Last" in keep_option else False
            exist_cols = [c for c in subset_cols if c in df.columns]
            df_clean = df.drop_duplicates(keep=keep, subset=exist_cols if exist_cols else None)
            df_preview = df_clean
            if preview_dup:
                st.write("Preview of DataFrame after Duplicate Removal:")
                st.dataframe(df_preview.head())
            if apply_dup:
                st.session_state['data'] = df_preview
                st.success("Duplicate removal applied!")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info(f"Found {dup_count} duplicate rows." if dup_count > 0 else "No duplicate rows found.")

with tab4:
    st.subheader("Standardisation & Normalization", divider=True)
    st.caption("Standardisation & Normalization are techniques used to transform the data into a common scale.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    select_numeric_cols = st.multiselect(
        "Select numeric columns for scaling",
        options=numeric_cols,
        default=None
    )
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
    preview_scale = st.button("Preview", key="preview_scale")
    apply_scale = st.button("Apply", key="apply_scale")
    df_preview = None

    if select_numeric_cols:
        missing_cols = [col for col in select_numeric_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Selected columns not in DataFrame: {missing_cols}")
            st.stop()
        X = df[select_numeric_cols] if len(select_numeric_cols) > 1 else df[[select_numeric_cols[0]]]
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        if preview_scale or apply_scale:
            try:
                if select_scale == "None":
                    X_train_scaled = X_train.copy()
                    X_test_scaled = X_test.copy()
                else:
                    scaler = scaler_dict[select_scale]
                    scaler.fit(X_train)
                    X_train_scaled = pd.DataFrame(
                        scaler.transform(X_train),
                        columns=X_train.columns,
                        index=X_train.index
                    )
                    X_test_scaled = pd.DataFrame(
                        scaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index
                    )
                X_scaled = pd.concat([X_train_scaled, X_test_scaled]).sort_index()
                obj_cols = df.select_dtypes(exclude=[np.number]).columns
                numeric_idx = X_scaled.index
                obj_data_aligned = df.loc[numeric_idx, obj_cols]
                final_scaled_df = pd.concat([X_scaled, obj_data_aligned], axis=1)
                # Only reorder to columns that are still present
                original_cols = [col for col in df.columns if col in final_scaled_df.columns]
                final_scaled_df = final_scaled_df[original_cols]
                df_preview = final_scaled_df
                if preview_scale:
                    st.write("Preview of Scaled DataFrame (with object columns):")
                    st.dataframe(df_preview.head())
                if apply_scale:
                    st.session_state['scaled_data'] = df_preview
                    st.session_state['data'] = df_preview
                    st.success("Scaled data saved!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("Please select at least one numeric column for scaling.")

with tab5:
    st.subheader("Encoding Categorical Data", divider=True)
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    select_encode = st.selectbox(
        "Select encoding method",
        ["One-hot encoding", "Label encoding", "Ordinal encoding", "Binning", "None"]
    )
    df_preview = None

    if select_encode == "None":
        st.write("No encoding method selected.")
    elif select_encode == "Label encoding":
        label_cols = st.multiselect("Select columns for label encoding", options=obj_cols, default=[])
        preview_label = st.button("Preview", key="preview_label")
        apply_label = st.button("Apply", key="apply_label")
        if label_cols and (preview_label or apply_label):
            df_label = df.copy()
            le = LabelEncoder()
            for col in label_cols:
                if col in df_label.columns:
                    df_label[col] = le.fit_transform(df_label[col].astype(str))
            df_preview = df_label
            if preview_label:
                st.write("Preview of Label-encoded DataFrame:")
                st.dataframe(df_preview.head())
            if apply_label:
                st.session_state['data'] = df_preview
                st.success("Label-encoded data saved!")
                st.rerun()
    elif select_encode == "Ordinal encoding":
        ordinal_cols = st.multiselect("Select columns for ordinal encoding", options=obj_cols, default=[])
        preview_ordinal = st.button("Preview", key="preview_ordinal")
        apply_ordinal = st.button("Apply", key="apply_ordinal")
        if ordinal_cols and (preview_ordinal or apply_ordinal):
            df_ordinal = df.copy()
            oe = OrdinalEncoder()
            for col in ordinal_cols:
                if col in df_ordinal.columns:
                    df_ordinal[col] = oe.fit_transform(df_ordinal[col].values.reshape(-1, 1))
            df_preview = df_ordinal
            if preview_ordinal:
                st.write("Preview of Ordinal-encoded DataFrame:")
                st.dataframe(df_preview.head())
            if apply_ordinal:
                st.session_state['data'] = df_preview
                st.success("Ordinal-encoded data saved!")
                st.rerun()
    elif select_encode == "Binning":
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        select_numeric = st.multiselect("Select numeric columns for binning", options=numeric_cols, default=[])
        strategy_options_list = ["quantile", "uniform", "kmeans", "None"]
        strategy = st.selectbox("Select strategy", options=strategy_options_list, index=strategy_options_list.index("None"))
        n_bins = st.number_input("Number of bins", min_value=2, max_value=10, value=5)
        preview_bin = st.button("Preview", key="preview_bin")
        apply_bin = st.button("Apply", key="apply_bin")
        if select_numeric and strategy != "None" and (preview_bin or apply_bin):
            df_processed = df.copy()
            try:
                exist_cols = [col for col in select_numeric if col in df_processed.columns]
                if exist_cols:
                    kbin = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
                    transformed = kbin.fit_transform(df_processed[exist_cols])
                    df_processed[exist_cols] = pd.DataFrame(transformed, columns=exist_cols, index=df_processed.index)
                    for i, col in enumerate(exist_cols):
                        edges = kbin.bin_edges_[i]
                        intervals = [
                            f"[{edges[j]:.2f}–{edges[j + 1]:.2f})"
                            for j in range(len(edges) - 1)
                        ]
                        df_processed[col + "_range"] = df_processed[col].apply(
                            lambda x: intervals[int(x)] if not pd.isna(x) else None
                        )
                    df_preview = df_processed
                    if preview_bin:
                        st.write("Preview of Binned DataFrame:")
                        st.dataframe(df_preview.head())
                    if apply_bin:
                        st.session_state['data'] = df_preview
                        st.success("Binned data saved!")
                        st.rerun()
                else:
                    st.warning("No selected numeric columns exist in DataFrame for binning.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:  # One-hot encoding
        selected_cols = st.multiselect("Select columns for one-hot encoding", options=obj_cols, default=[])
        preview_onehot = st.button("Preview", key="preview_onehot")
        apply_onehot = st.button("Apply", key="apply_onehot")
        if selected_cols and (preview_onehot or apply_onehot):
            exist_cols = [col for col in selected_cols if col in df.columns]
            if exist_cols:
                df_encoded = pd.get_dummies(df, columns=exist_cols)
                df_preview = df_encoded
                if preview_onehot:
                    st.write("Preview of One-hot Encoded DataFrame:")
                    st.dataframe(df_preview.head())
                if apply_onehot:
                    st.session_state['data'] = df_preview
                    st.success("One-hot encoded data saved!")
                    st.rerun()
            else:
                st.warning("No selected object columns found for one-hot encoding.")
