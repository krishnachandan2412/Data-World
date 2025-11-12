import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Page configuration
st.set_page_config(page_title="Data Visualization", page_icon="📊", layout="centered",initial_sidebar_state="expanded")
st.header("📊:blue[ Data Visualization]",divider=True)

# Helper functions
def get_col_type(df, col):
    if pd.api.types.is_numeric_dtype(df[col]):
        return "num"
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        return "date"
    else:
        return "cat"

def get_colname(opt):
    return opt.split(' (')[0] if " (" in opt else opt

def col_opts(df):
    return [f"{c} ({get_col_type(df, c)})" for c in df.columns]

graph_types = [
    'line', 'bar', 'scatter', 'pie', 'sunburst',
    'distplot', 'boxplot', 'violin', 'bubble',
    'treemap', 'histogram', 'heatmap', 'area', 'funnel', 'scatter_matrix'
]
chart_type_map = {
    'distplot': 'Statistical', 'boxplot': 'Statistical', 'violin': 'Statistical', 'histogram': 'Statistical',
    'heatmap': 'Statistical',
    'line': 'Basic', 'bar': 'Basic', 'scatter': 'Basic', 'bubble': 'Basic',
    'pie': 'Basic', 'sunburst': 'Basic', 'treemap': 'Basic', 'area': 'Basic', 'funnel': 'Basic',
    'scatter_matrix': 'Basic'
}

tab1, tab2, tab3 = st.tabs([
    'Import Session Data',
    'Groupby Data',
    'Import Data'
])

with tab1:
    st.subheader("Data from session")
    df = st.session_state.get('data', None)
    if df is None:
        st.warning("No data found. Please load data first (go to Groupby tab and upload a file).")
    else:
        st.success(" Session DataFrame loaded!")
        st.dataframe(df.head())
        result = df

        st.subheader("Visualization of groupby")
        opts = col_opts(result)
        graphs_selected = st.multiselect('Main chart', options=graph_types, key='main_chart_multiselect_tab1')
        if graphs_selected:
            st.caption(f"Selected chart types: {', '.join([chart_type_map.get(g, 'Other') for g in graphs_selected])}")
            for graphs in graphs_selected:
                st.markdown(f"### {graphs.capitalize()} Chart")
                try:
                    fig = None
                    if graphs == 'line':
                        x_axis = st.selectbox(f'Line X axis [{graphs}]', opts, key=f"line_x_{graphs}_tab1")
                        y_axis = st.selectbox(f'Line Y axis (num) [{graphs}]', opts, key=f"line_y_{graphs}_tab1")
                        color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"line_color_{graphs}_tab1")
                        fig = px.line(
                            result,
                            x=get_colname(x_axis), y=get_colname(y_axis),
                            color=get_colname(color) if color else None, markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'bar':
                        x_axis = st.selectbox(f'Bar X axis [{graphs}]', opts, key=f"bar_x_{graphs}_tab1")
                        y_axis = st.selectbox(f'Bar Y axis (num) [{graphs}]', opts, key=f"bar_y_{graphs}_tab1")
                        color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"bar_color_{graphs}_tab1")
                        facet_col = st.selectbox(f'Facet [{graphs}]', [None] + opts, key=f"bar_facet_{graphs}_tab1")
                        fig = px.bar(
                            result, x=get_colname(x_axis), y=get_colname(y_axis),
                            color=get_colname(color) if color else None,
                            facet_col=get_colname(facet_col) if facet_col else None,
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs in ['scatter', 'bubble']:
                        x_axis = st.selectbox(f'X axis (num) [{graphs}]', opts, key=f"scatter_x_{graphs}_tab1")
                        y_axis = st.selectbox(f'Y axis (num) [{graphs}]', opts, key=f"scatter_y_{graphs}_tab1")
                        color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"scatter_color_{graphs}_tab1")
                        size = st.selectbox(f'Size (num only) [{graphs}]', [None] + opts, key=f"scatter_size_{graphs}_tab1")
                        size_col = get_colname(size) if size else None
                        if size_col:
                            result[size_col] = pd.to_numeric(result[size_col], errors='coerce').fillna(1)
                        fig = px.scatter(
                            result,
                            x=get_colname(x_axis), y=get_colname(y_axis),
                            color=get_colname(color) if color else None,
                            size=size_col if size_col else None
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'pie':
                        values = st.selectbox(f'Pie Values (num) [{graphs}]', opts, key=f"pie_values_{graphs}_tab1")
                        names = st.selectbox(f'Pie Names (cat) [{graphs}]', opts, key=f"pie_names_{graphs}_tab1")
                        fig = px.pie(result, values=get_colname(values), names=get_colname(names))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs in ['sunburst', 'treemap']:
                        path = st.multiselect(f"{graphs.capitalize()} path (cat) [{graphs}]", opts,
                                              key=f"{graphs}_path_{graphs}_tab1")
                        value_col = st.selectbox(f"{graphs.capitalize()} values (num) [{graphs}]", opts,
                                                 key=f"{graphs}_value_{graphs}_tab1")
                        fig_func = px.sunburst if graphs == 'sunburst' else px.treemap
                        fig = fig_func(
                            result,
                            path=[get_colname(p) for p in path],
                            values=get_colname(value_col)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'histogram':
                        x_axis = st.selectbox(f'Histogram X axis (num) [{graphs}]', opts, key=f"hist_x_{graphs}_tab1")
                        fig = px.histogram(result, x=get_colname(x_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'boxplot':
                        x_axis = st.selectbox(f'Boxplot X axis (num) [{graphs}]', opts, key=f"box_x_{graphs}_tab1")
                        fig = px.box(result, x=get_colname(x_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'violin':
                        x_axis = st.selectbox(f'Violin X axis (num) [{graphs}]', opts, key=f"violin_x_{graphs}_tab1")
                        fig = px.violin(result, x=get_colname(x_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'distplot':
                        numeric_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                        if numeric_cols:
                            x_axis = st.selectbox(f'Distplot column (num only) [{graphs}]', numeric_cols,
                                                  key=f"distplot_x_{graphs}_tab1")
                            hist_data = [result[x_axis].dropna().values]
                            group_labels = [x_axis]
                            fig = ff.create_distplot(hist_data, group_labels, bin_size=10)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("No numeric columns available for distplot.")

                    elif graphs == 'heatmap':
                        num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                        x_axis = st.selectbox(f'Heatmap X (num) [{graphs}]', num_cols, key=f"heatmap_x_{graphs}_tab1")
                        y_axis = st.selectbox(f'Heatmap Y (num) [{graphs}]', num_cols, key=f"heatmap_y_{graphs}_tab1")
                        fig = px.density_heatmap(result, x=x_axis, y=y_axis)
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'area':
                        x_axis = st.selectbox(f'Area X axis (cat or num) [{graphs}]', opts, key=f"area_x_{graphs}_tab1")
                        y_axis = st.selectbox(f'Area Y axis (num) [{graphs}]', opts, key=f"area_y_{graphs}_tab1")
                        fig = px.area(result, x=get_colname(x_axis), y=get_colname(y_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'funnel':
                        x_axis = st.selectbox(f'Funnel X axis (cat) [{graphs}]', opts, key=f"funnel_x_{graphs}_tab1")
                        y_axis = st.selectbox(f'Funnel Y axis (num) [{graphs}]', opts, key=f"funnel_y_{graphs}_tab1")
                        fig = px.funnel(result, x=get_colname(x_axis), y=get_colname(y_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'scatter_matrix':
                        num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                        if len(num_cols) >= 2:
                            fig = px.scatter_matrix(result, dimensions=num_cols)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Scatter Matrix requires at least 2 numerical columns.")
                    else:
                        st.warning(f'Select a proper graph type: {graphs}')
                except Exception as e:
                    st.error(f"Chart Error [{graphs}]: {str(e)}")
        else:
            st.info("Select at least one chart type from the multiselect to visualize.")

with tab2:
    st.subheader("Data from groupby")
    uploaded_file = st.file_uploader("Upload Groupby file", type="csv", key="groupby_upload")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.success(" DataFrame loaded!")
        st.dataframe(df.head())
        result = df

        st.subheader("Visualization of groupby")
        opts = col_opts(result)
        graphs_selected = st.multiselect('Main chart', options=graph_types, key='main_chart_multiselect_tab2')
        if graphs_selected:
            st.caption(f"Selected chart types: {', '.join([chart_type_map.get(g, 'Other') for g in graphs_selected])}")
            for graphs in graphs_selected:
                st.markdown(f"### {graphs.capitalize()} Chart")
                try:
                    fig = None
                    if graphs == 'line':
                        x_axis = st.selectbox(f'Line X axis [{graphs}]', opts, key=f"line_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Line Y axis (num) [{graphs}]', opts, key=f"line_y_{graphs}_tab2")
                        color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"line_color_{graphs}_tab2")
                        fig = px.line(
                            result,
                            x=get_colname(x_axis), y=get_colname(y_axis),
                            color=get_colname(color) if color else None, markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'bar':
                        x_axis = st.selectbox(f'Bar X axis [{graphs}]', opts, key=f"bar_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Bar Y axis (num) [{graphs}]', opts, key=f"bar_y_{graphs}_tab2")
                        color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"bar_color_{graphs}_tab2")
                        facet_col = st.selectbox(f'Facet [{graphs}]', [None] + opts, key=f"bar_facet_{graphs}_tab2")
                        fig = px.bar(
                            result, x=get_colname(x_axis), y=get_colname(y_axis),
                            color=get_colname(color) if color else None,
                            facet_col=get_colname(facet_col) if facet_col else None,
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs in ['scatter', 'bubble']:
                        x_axis = st.selectbox(f'X axis (num) [{graphs}]', opts, key=f"scatter_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Y axis (num) [{graphs}]', opts, key=f"scatter_y_{graphs}_tab2")
                        color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"scatter_color_{graphs}_tab2")
                        size = st.selectbox(f'Size (num only) [{graphs}]', [None] + opts, key=f"scatter_size_{graphs}_tab2")
                        size_col = get_colname(size) if size else None
                        if size_col:
                            result[size_col] = pd.to_numeric(result[size_col], errors='coerce').fillna(1)
                        fig = px.scatter(
                            result,
                            x=get_colname(x_axis), y=get_colname(y_axis),
                            color=get_colname(color) if color else None,
                            size=size_col if size_col else None
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'pie':
                        values = st.selectbox(f'Pie Values (num) [{graphs}]', opts, key=f"pie_values_{graphs}_tab2")
                        names = st.selectbox(f'Pie Names (cat) [{graphs}]', opts, key=f"pie_names_{graphs}_tab2")
                        fig = px.pie(result, values=get_colname(values), names=get_colname(names))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs in ['sunburst', 'treemap']:
                        path = st.multiselect(f"{graphs.capitalize()} path (cat) [{graphs}]", opts,
                                              key=f"{graphs}_path_{graphs}_tab2")
                        value_col = st.selectbox(f"{graphs.capitalize()} values (num) [{graphs}]", opts,
                                                 key=f"{graphs}_value_{graphs}_tab2")
                        fig_func = px.sunburst if graphs == 'sunburst' else px.treemap
                        fig = fig_func(
                            result,
                            path=[get_colname(p) for p in path],
                            values=get_colname(value_col)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'histogram':
                        x_axis = st.selectbox(f'Histogram X axis (num) [{graphs}]', opts, key=f"hist_x_{graphs}_tab2")
                        fig = px.histogram(result, x=get_colname(x_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'boxplot':
                        x_axis = st.selectbox(f'Boxplot X axis (num) [{graphs}]', opts, key=f"box_x_{graphs}_tab2")
                        fig = px.box(result, x=get_colname(x_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'violin':
                        x_axis = st.selectbox(f'Violin X axis (num) [{graphs}]', opts, key=f"violin_x_{graphs}_tab2")
                        fig = px.violin(result, x=get_colname(x_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'distplot':
                        numeric_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                        if numeric_cols:
                            x_axis = st.selectbox(f'Distplot column (num only) [{graphs}]', numeric_cols,
                                                  key=f"distplot_x_{graphs}_tab2")
                            hist_data = [result[x_axis].dropna().values]
                            group_labels = [x_axis]
                            fig = ff.create_distplot(hist_data, group_labels, bin_size=10)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("No numeric columns available for distplot.")

                    elif graphs == 'heatmap':
                        num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                        x_axis = st.selectbox(f'Heatmap X (num) [{graphs}]', num_cols, key=f"heatmap_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Heatmap Y (num) [{graphs}]', num_cols, key=f"heatmap_y_{graphs}_tab2")
                        fig = px.density_heatmap(result, x=x_axis, y=y_axis)
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'area':
                        x_axis = st.selectbox(f'Area X axis (cat or num) [{graphs}]', opts, key=f"area_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Area Y axis (num) [{graphs}]', opts, key=f"area_y_{graphs}_tab2")
                        fig = px.area(result, x=get_colname(x_axis), y=get_colname(y_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'funnel':
                        x_axis = st.selectbox(f'Funnel X axis (cat) [{graphs}]', opts, key=f"funnel_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Funnel Y axis (num) [{graphs}]', opts, key=f"funnel_y_{graphs}_tab2")
                        fig = px.funnel(result, x=get_colname(x_axis), y=get_colname(y_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'scatter_matrix':
                        num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                        if len(num_cols) >= 2:
                            fig = px.scatter_matrix(result, dimensions=num_cols)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Scatter Matrix requires at least 2 numerical columns.")
                    else:
                        st.warning(f'Select a proper graph type: {graphs}')
                except Exception as e:
                    st.error(f"Chart Error [{graphs}]: {str(e)}")
        else:
            st.info("Select at least one chart type from the multiselect to visualize.")
    else:
        st.warning("No data found for groupby. Please upload a file.")

with tab3:
    st.subheader("Import any CSV file")
    uploaded_file = st.file_uploader("Upload Data file", type="csv", key="other_upload")
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        df = pd.read_csv(uploaded_file)
        st.write("Data Loaded!")
        st.dataframe(df)
        st.subheader("Visualization of groupby")
        opts = col_opts(result)
        graphs_selected = st.multiselect('Main chart', options=graph_types, key='main_chart_multiselect_tab2')
        if graphs_selected:
            st.caption(f"Selected chart types: {', '.join([chart_type_map.get(g, 'Other') for g in graphs_selected])}")
            for graphs in graphs_selected:
                st.markdown(f"### {graphs.capitalize()} Chart")
                try:
                    fig = None
                    if graphs == 'line':
                        x_axis = st.selectbox(f'Line X axis [{graphs}]', opts, key=f"line_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Line Y axis (num) [{graphs}]', opts, key=f"line_y_{graphs}_tab2")
                        color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"line_color_{graphs}_tab2")
                        fig = px.line(
                            result,
                            x=get_colname(x_axis), y=get_colname(y_axis),
                            color=get_colname(color) if color else None, markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'bar':
                        x_axis = st.selectbox(f'Bar X axis [{graphs}]', opts, key=f"bar_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Bar Y axis (num) [{graphs}]', opts, key=f"bar_y_{graphs}_tab2")
                        color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"bar_color_{graphs}_tab2")
                        facet_col = st.selectbox(f'Facet [{graphs}]', [None] + opts, key=f"bar_facet_{graphs}_tab2")
                        fig = px.bar(
                            result, x=get_colname(x_axis), y=get_colname(y_axis),
                            color=get_colname(color) if color else None,
                            facet_col=get_colname(facet_col) if facet_col else None,
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs in ['scatter', 'bubble']:
                        x_axis = st.selectbox(f'X axis (num) [{graphs}]', opts, key=f"scatter_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Y axis (num) [{graphs}]', opts, key=f"scatter_y_{graphs}_tab2")
                        color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"scatter_color_{graphs}_tab2")
                        size = st.selectbox(f'Size (num only) [{graphs}]', [None] + opts,
                                            key=f"scatter_size_{graphs}_tab2")
                        size_col = get_colname(size) if size else None
                        if size_col:
                            result[size_col] = pd.to_numeric(result[size_col], errors='coerce').fillna(1)
                        fig = px.scatter(
                            result,
                            x=get_colname(x_axis), y=get_colname(y_axis),
                            color=get_colname(color) if color else None,
                            size=size_col if size_col else None
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'pie':
                        values = st.selectbox(f'Pie Values (num) [{graphs}]', opts, key=f"pie_values_{graphs}_tab2")
                        names = st.selectbox(f'Pie Names (cat) [{graphs}]', opts, key=f"pie_names_{graphs}_tab2")
                        fig = px.pie(result, values=get_colname(values), names=get_colname(names))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs in ['sunburst', 'treemap']:
                        path = st.multiselect(f"{graphs.capitalize()} path (cat) [{graphs}]", opts,
                                              key=f"{graphs}_path_{graphs}_tab2")
                        value_col = st.selectbox(f"{graphs.capitalize()} values (num) [{graphs}]", opts,
                                                 key=f"{graphs}_value_{graphs}_tab2")
                        fig_func = px.sunburst if graphs == 'sunburst' else px.treemap
                        fig = fig_func(
                            result,
                            path=[get_colname(p) for p in path],
                            values=get_colname(value_col)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'histogram':
                        x_axis = st.selectbox(f'Histogram X axis (num) [{graphs}]', opts, key=f"hist_x_{graphs}_tab2")
                        fig = px.histogram(result, x=get_colname(x_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'boxplot':
                        x_axis = st.selectbox(f'Boxplot X axis (num) [{graphs}]', opts, key=f"box_x_{graphs}_tab2")
                        fig = px.box(result, x=get_colname(x_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'violin':
                        x_axis = st.selectbox(f'Violin X axis (num) [{graphs}]', opts, key=f"violin_x_{graphs}_tab2")
                        fig = px.violin(result, x=get_colname(x_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'distplot':
                        numeric_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                        if numeric_cols:
                            x_axis = st.selectbox(f'Distplot column (num only) [{graphs}]', numeric_cols,
                                                  key=f"distplot_x_{graphs}_tab2")
                            hist_data = [result[x_axis].dropna().values]
                            group_labels = [x_axis]
                            fig = ff.create_distplot(hist_data, group_labels, bin_size=10)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("No numeric columns available for distplot.")

                    elif graphs == 'heatmap':
                        num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                        x_axis = st.selectbox(f'Heatmap X (num) [{graphs}]', num_cols, key=f"heatmap_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Heatmap Y (num) [{graphs}]', num_cols, key=f"heatmap_y_{graphs}_tab2")
                        fig = px.density_heatmap(result, x=x_axis, y=y_axis)
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'area':
                        x_axis = st.selectbox(f'Area X axis (cat or num) [{graphs}]', opts, key=f"area_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Area Y axis (num) [{graphs}]', opts, key=f"area_y_{graphs}_tab2")
                        fig = px.area(result, x=get_colname(x_axis), y=get_colname(y_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'funnel':
                        x_axis = st.selectbox(f'Funnel X axis (cat) [{graphs}]', opts, key=f"funnel_x_{graphs}_tab2")
                        y_axis = st.selectbox(f'Funnel Y axis (num) [{graphs}]', opts, key=f"funnel_y_{graphs}_tab2")
                        fig = px.funnel(result, x=get_colname(x_axis), y=get_colname(y_axis))
                        st.plotly_chart(fig, use_container_width=True)

                    elif graphs == 'scatter_matrix':
                        num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                        if len(num_cols) >= 2:
                            fig = px.scatter_matrix(result, dimensions=num_cols)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Scatter Matrix requires at least 2 numerical columns.")
                    else:
                        st.warning(f'Select a proper graph type: {graphs}')
                except Exception as e:
                    st.error(f"Chart Error [{graphs}]: {str(e)}")
        else:
            st.info("Select at least one chart type from the multiselect to visualize.")
    else:
        st.info("Please upload a CSV file to see its contents.")
