import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import palettable.colorbrewer.diverging as pld

# Constant
PALETTE = pld.Spectral_4_r  # _r if you want to reverse the color sequence
CMAP = PALETTE.mpl_colormap     # .mpl_colormap attribute is a continuous, interpolated map


def main():
    df = sns.load_dataset('titanic')
    st.title('Titanic Dashboard')

    st.subheader('Dataset')
    st.dataframe(df)

    st.subheader('Data Numerical Statistic')
    st.dataframe(df.describe())

    st.subheader('Data Visualization with respect to Survived')
    left_column, right_column = st.columns(2)
    with left_column:
        'Numerical Plot'
        num_feat = st.selectbox('Select Numerical Feature', df.select_dtypes('number').columns)
        fig = px.histogram(df, x=num_feat, color='survived')
        st.plotly_chart(fig, use_container_width=True)
    with right_column:
        'Categorical column'
        cat_feat = st.selectbox('Select Categorical Feature', df.select_dtypes(exclude='number').columns)
        fig = px.histogram(df, x=cat_feat, color='survived')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Map nearby Stony Brook')
    number_of_points = 100
    coordinate_dimension = 2
    distance = [1 / 50, 1 / 50]
    origin_position = [40.9, -73.1]
    map_data = pd.DataFrame(
        np.random.randn(number_of_points, coordinate_dimension) * distance + origin_position,
        columns=['lat', 'lon'])
    st.map(map_data, zoom=10)  # 40.89320673626953, -73.11882380033349, 22.7, 120.3

    st.subheader('My bar chart')
    bar_chart()


def bar_chart():
    fig, ax = plt.subplots()
    colors = ["#69b3a2", "#4374B3"]
    # sns.set_palette(sns.color_palette(colors))

    samples = ['NbAl\npristine', 'Sc/NbAl\npristine', 'Sc/NbAl\n900C60M', 'Sc/NbAl\n1100C60M']
    element_A_at = [53.00, 51.23, 50.95, 50.52]
    elememt_B_at = [47.00, 48.77, 49.05, 49.48]

    x = np.arange(len(samples))
    bar_width = 0.4
    color_idx = np.linspace(0, 1, len(samples))

    # Plotting
    bar_A = plt.bar(x, element_A_at, bar_width, color=CMAP(color_idx[0]), label='Nb', edgecolor='white')
    bar_B = plt.bar(x + bar_width, elememt_B_at, bar_width, color=CMAP(color_idx[1]), label='Al', edgecolor='white')

    # Texts
    for index, bar in enumerate(bar_A):
        height_A = bar.get_height()
        height_B = 100 - height_A

        ax.text(bar.get_x() + bar.get_width() / 2, 1.05 * height_A,
                f'{element_A_at[index]}',
                ha='center', va='bottom', rotation=0, fontsize=12)
        ax.text(bar.get_x() + bar.get_width() / 2 + bar_width, 1.05 * height_B,
                f'{elememt_B_at[index]}',
                ha='center', va='bottom', rotation=0, fontsize=12)

    # Frame linewidth
    spineline = ['left', 'right', 'top', 'bottom']
    for direction in spineline:
        ax.spines[direction].set_linewidth('2')

    # Formatting
    plt.yticks(fontsize=14)
    ax.tick_params(width=2)
    plt.xticks(x + bar_width / 2, samples, fontsize=14)
    plt.ylabel('Atomic percentage', fontsize=14)
    plt.ylim(0, 80)
    plt.title('')
    plt.legend(fontsize=14)
    plt.tight_layout()
    # plt.show()
    st.pyplot(fig)


if __name__ == '__main__':
    main()