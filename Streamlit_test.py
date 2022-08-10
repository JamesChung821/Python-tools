import streamlit as st
import pandas as pd
# import plotly.express as px
import seaborn as sns
import numpy as np


def main():
    df = sns.load_dataset('titanic')
    st.title('Titanic Dashboard')
    st.subheader('Dataset')
    st.dataframe(df)
    st.subheader('Data Numerical Statistic')
    st.dataframe(df.describe())
    st.subheader('Data Visualization with respect to Survived')
    left_column, right_column = st.columns(2)
    # with left_column:
    #    'Numerical Plot'
    #     num_feat = st.selectbox(
    #    'Select Numerical Feature', df.select_dtypes('number').columns)
    #     fig = px.histogram(df, x = num_feat, color = 'survived')
    #     st.plotly_chart(fig, use_container_width=True)
    # with right_column:
    #    'Categorical column'
    #     cat_feat = st.selectbox(
    #     'Select Categorical Feature', df.select_dtypes(exclude =   'number').columns)
    #     fig = px.histogram(df, x =cat_feat, color = 'survived' )
    # st.plotly_chart(fig, use_container_width=True)
    number_of_points = 100
    coordinate_dimension = 2
    distance = [1 / 50, 1 / 50]
    origin_position = [40.9, -73.1]
    map_data = pd.DataFrame(
        np.random.randn(number_of_points, coordinate_dimension) * distance + origin_position,
        columns=['lat', 'lon'])
    st.map(map_data, zoom=10)  # 40.89320673626953, -73.11882380033349, 22.7, 120.3


if __name__ == '__main__':
    main()