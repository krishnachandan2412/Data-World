import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

st.set_page_config(page_title="Feature Engineering", layout="centered",page_icon="🎯",initial_sidebar_state="expanded")
st.header('🎯:blue[Feature Engineering]',divider=True)
st.subheader("Here you can create new features from existing ones",divider=True)

if 'data' not in st.session_state or st.session_state['data'] is None:
    st.warning("No data found. Please load data first.")
    st.stop()

df = st.session_state['data']
st.success("✅ Session DataFrame loaded!")
st.dataframe(df)  # Display a sample of the dataframe

# ---- CUSTOM ROW MEASURE ----
st.subheader(':gray[Creating new measures]', divider='rainbow')
st.markdown("####  Add Custom (Row-level) Measure to Source Data")
with st.form("custom_row_measure_form"):
    row_measure_name = st.text_input("Name for new row-level measure", value="")
    st.info("Ex: np.where(Num_Missed_Payments > 4, 'NPA', 'Performer')")
    row_measure_formula = st.text_input("Formula, use df columns. ", value="")
    row_submitted = st.form_submit_button("Add Row-level Measure")
    if row_submitted and row_measure_name and row_measure_formula:
        try:
            safe_ns = {"np": np, "pd": pd}
            for c in df.columns:
                safe_ns[c] = df[c]
            df[row_measure_name] = eval(row_measure_formula, {}, safe_ns)
            st.success(f"Row-level custom measure '{row_measure_name}' added to df!")
            st.rerun()
        except Exception as e:
            st.error(f"Error in row-level formula: {e}")

# ---- GROUPBY & DASHBOARD ----

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

obj_ops = ['count', 'nunique', 'first', 'last']  # Operations for object/cat/date
num_ops = ['sum', 'max', 'min', 'mean', 'median', 'count', 'nunique', 'first', 'last']

st.subheader(':gray[Groupby : Simplify your data analysis]', divider=True)
st.write('The groupby lets you summarize data by specific categories and groups')

col1, col2, col3 = st.columns(3)
with col1:
    groupby_cols = st.multiselect(
        'Choose your column to groupby',
        options=col_opts(df)
    )
with col2:
    operation_col = st.selectbox(
        'Choose column for operation',
        options=col_opts(df)
    )
op_col_selection = get_colname(operation_col)
op_col_type = get_col_type(df, op_col_selection)

with col3:
    if op_col_type == 'num':
        operation = st.selectbox('Choose operation', num_ops)
    else:
        operation = st.selectbox('Choose operation', obj_ops)

groupby_selection = [get_colname(e) for e in groupby_cols] if groupby_cols else []

if groupby_selection and op_col_selection:
    result = df.groupby(groupby_selection).agg(
        **{op_col_selection: (op_col_selection, operation)}
    ).reset_index()
    st.dataframe(result)
    # download result
    st.download_button(
        label="Download result",
        data=result.to_csv(index=False).encode('utf-8'),
        file_name='groupby_result.csv',
        mime='text/csv'
    )

    # ---- CHARTS ----
    # ... previous processing code ...

    st.subheader("Visualization of groupby")
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
    opts = col_opts(result)

    graphs_selected = st.multiselect('Main chart', options=graph_types)
    if graphs_selected:
        st.caption(f"Selected chart types: {', '.join([chart_type_map.get(g, 'Other') for g in graphs_selected])}")
        # loop for multiple charts
        for graphs in graphs_selected:
            st.markdown(f"### {graphs.capitalize()} Chart")
            try:
                fig = None
                if graphs == 'line':
                    x_axis = st.selectbox(f'Line X axis [{graphs}]', opts, key=f"line_x_{graphs}")
                    y_axis = st.selectbox(f'Line Y axis (num) [{graphs}]', opts, key=f"line_y_{graphs}")
                    color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"line_color_{graphs}")
                    fig = px.line(
                        result,
                        x=get_colname(x_axis), y=get_colname(y_axis),
                        color=get_colname(color) if color else None, markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"line_{x_axis}_{y_axis}_{color}")

                elif graphs == 'bar':
                    x_axis = st.selectbox(f'Bar X axis [{graphs}]', opts, key=f"bar_x_{graphs}")
                    y_axis = st.selectbox(f'Bar Y axis (num) [{graphs}]', opts, key=f"bar_y_{graphs}")
                    color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"bar_color_{graphs}")
                    facet_col = st.selectbox(f'Facet [{graphs}]', [None] + opts, key=f"bar_facet_{graphs}")
                    fig = px.bar(
                        result, x=get_colname(x_axis), y=get_colname(y_axis),
                        color=get_colname(color) if color else None,
                        facet_col=get_colname(facet_col) if facet_col else None,
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"bar_{x_axis}_{y_axis}_{color}_{facet_col}")

                elif graphs in ['scatter', 'bubble']:
                    x_axis = st.selectbox(f'X axis (num) [{graphs}]', opts, key=f"scatter_x_{graphs}")
                    y_axis = st.selectbox(f'Y axis (num) [{graphs}]', opts, key=f"scatter_y_{graphs}")
                    color = st.selectbox(f'Color [{graphs}]', [None] + opts, key=f"scatter_color_{graphs}")
                    size = st.selectbox(f'Size (num only) [{graphs}]', [None] + opts, key=f"scatter_size_{graphs}")
                    size_col = get_colname(size) if size else None
                    if size_col:
                        result[size_col] = pd.to_numeric(result[size_col], errors='coerce').fillna(1)
                    fig = px.scatter(
                        result,
                        x=get_colname(x_axis), y=get_colname(y_axis),
                        color=get_colname(color) if color else None,
                        size=size_col if size_col else None
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"{graphs}_{x_axis}_{y_axis}_{color}_{size}")

                elif graphs == 'pie':
                    values = st.selectbox(f'Pie Values (num) [{graphs}]', opts, key=f"pie_values_{graphs}")
                    names = st.selectbox(f'Pie Names (cat) [{graphs}]', opts, key=f"pie_names_{graphs}")
                    fig = px.pie(result, values=get_colname(values), names=get_colname(names))
                    st.plotly_chart(fig, use_container_width=True, key=f"pie_{values}_{names}")

                elif graphs in ['sunburst', 'treemap']:
                    path = st.multiselect(f"{graphs.capitalize()} path (cat) [{graphs}]", opts,
                                          key=f"{graphs}_path_{graphs}")
                    value_col = st.selectbox(f"{graphs.capitalize()} values (num) [{graphs}]", opts,
                                             key=f"{graphs}_value_{graphs}")
                    fig_func = px.sunburst if graphs == 'sunburst' else px.treemap
                    fig = fig_func(
                        result,
                        path=[get_colname(p) for p in path],
                        values=get_colname(value_col)
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"{graphs}_{'_'.join(path)}_{value_col}")

                elif graphs == 'histogram':
                    x_axis = st.selectbox(f'Histogram X axis (num) [{graphs}]', opts, key=f"hist_x_{graphs}")
                    fig = px.histogram(result, x=get_colname(x_axis))
                    st.plotly_chart(fig, use_container_width=True, key=f"hist_{x_axis}")

                elif graphs == 'boxplot':
                    x_axis = st.selectbox(f'Boxplot X axis (num) [{graphs}]', opts, key=f"box_x_{graphs}")
                    fig = px.box(result, x=get_colname(x_axis))
                    st.plotly_chart(fig, use_container_width=True, key=f"box_{x_axis}")

                elif graphs == 'violin':
                    x_axis = st.selectbox(f'Violin X axis (num) [{graphs}]', opts, key=f"violin_x_{graphs}")
                    fig = px.violin(result, x=get_colname(x_axis))
                    st.plotly_chart(fig, use_container_width=True, key=f"violin_{x_axis}")

                elif graphs == 'distplot':
                    numeric_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                    if numeric_cols:
                        x_axis = st.selectbox(f'Distplot column (num only) [{graphs}]', numeric_cols,
                                              key=f"distplot_x_{graphs}")
                        hist_data = [result[x_axis].dropna().values]
                        group_labels = [x_axis]
                        fig = ff.create_distplot(hist_data, group_labels, bin_size=10)
                        st.plotly_chart(fig, use_container_width=True, key=f"distplot_{x_axis}")
                    else:
                        st.error("No numeric columns available for distplot.")

                elif graphs == 'heatmap':
                    num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                    x_axis = st.selectbox(f'Heatmap X (num) [{graphs}]', num_cols, key=f"heatmap_x_{graphs}")
                    y_axis = st.selectbox(f'Heatmap Y (num) [{graphs}]', num_cols, key=f"heatmap_y_{graphs}")
                    fig = px.density_heatmap(result, x=x_axis, y=y_axis)
                    st.plotly_chart(fig, use_container_width=True, key=f'heatmap_{x_axis}_{y_axis}')

                elif graphs == 'area':
                    x_axis = st.selectbox(f'Area X axis (cat or num) [{graphs}]', opts, key=f"area_x_{graphs}")
                    y_axis = st.selectbox(f'Area Y axis (num) [{graphs}]', opts, key=f"area_y_{graphs}")
                    fig = px.area(result, x=get_colname(x_axis), y=get_colname(y_axis))
                    st.plotly_chart(fig, use_container_width=True, key=f"area_{x_axis}_{y_axis}")

                elif graphs == 'funnel':
                    x_axis = st.selectbox(f'Funnel X axis (cat) [{graphs}]', opts, key=f"funnel_x_{graphs}")
                    y_axis = st.selectbox(f'Funnel Y axis (num) [{graphs}]', opts, key=f"funnel_y_{graphs}")
                    fig = px.funnel(result, x=get_colname(x_axis), y=get_colname(y_axis))
                    st.plotly_chart(fig, use_container_width=True, key=f"funnel_{x_axis}_{y_axis}")

                elif graphs == 'scatter_matrix':
                    num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                    if len(num_cols) >= 2:
                        fig = px.scatter_matrix(result, dimensions=num_cols)
                        st.plotly_chart(fig, use_container_width=True, key="scatter_matrix")
                    else:
                        st.error("Scatter Matrix requires at least 2 numerical columns.")

                else:
                    st.warning(f'Select a proper graph type: {graphs}')
            except Exception as e:
                st.error(f"Chart Error [{graphs}]: {str(e)}")
    else:
        st.info("Select at least one chart type from the multiselect to visualize.")

# Update session state at the end
st.session_state['data'] = df
