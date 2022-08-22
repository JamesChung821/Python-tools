import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import palettable.colorbrewer.diverging as pld
import streamlit_theme as stt
import time
import random
from collections import defaultdict
from sklearn import preprocessing, model_selection, tree, decomposition, ensemble, cluster, neighbors
from PIL import Image
import requests
from io import BytesIO

# Constant
PALETTE = pld.Spectral_4_r  # _r if you want to reverse the color sequence
CMAP = PALETTE.mpl_colormap     # .mpl_colormap attribute is a continuous, interpolated map

# Pre-step 2: Decide the model parameters
PCA = [2] 	# Principle Component Analysis
DEGREES = [1]		# PolynomialFeatures
DATA_SIZE = 800
N_CLUSTERS = DATA_SIZE//100
VISUALIZATION = False
AGE = ['11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '70+']
GENDER = ['Male', 'Female', 'Third gender']
GOAL = ['Date', 'Business', 'Friends', 'Family', 'Alone', 'Other']
HOURS = ['Brunch', 'Lunch', 'Dinner', 'Anytime']
DISTANCE = ['0-25', '26-50', '51-100', 'idc']


def main():
    # df = sns.load_dataset('titanic')
    df = synthetic_dataset()
    stt.set_theme({'primary': '#1b3388'})    # Useless QQ
    st.title('Restaurant AI Selector')
    st.subheader('Original Dataset')
    st.dataframe(df)

    st.subheader('Train Dataset')
    # st.dataframe(df.describe())
    train_data, label_data, training_dataframe = data_preprocess(synthetic_dataset(), mode='Train')  # Extract data and labels
    st.dataframe(train_data)

    st.subheader("User's Input")
    test_data = user_input_feature()
    column_dimension = test_data.shape[1]
    st.dataframe(test_data)

    st.subheader("Test Dataset")
    test_data = data_preprocess(test_data, mode='Test', training_dataframe=training_dataframe)
    st.dataframe(test_data)

    standardizier = preprocessing.StandardScaler()  # Call a standardization object
    x_train = standardizier.fit_transform(train_data)  # Do the standardization
    x_test = standardizier.transform(test_data)  # Do 'transform' only on testing data

    pca = decomposition.PCA(n_components=column_dimension)  # Data compression
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    k_means = cluster.KMeans(n_clusters=N_CLUSTERS)  # <-------------------------------- number of clusters
    k_means.fit(x_train_pca)  # No labels
    # print('Cluster center data:', k_means.cluster_centers_)
    print('====================================')
    print('Cluster label')
    print(k_means.labels_)
    # print(k_means.get_params())

    predict_label = k_means.predict(x_test_pca)
    print('====================================')
    print('Predict label')
    print(predict_label)
    print('Data format of each data')
    print(x_test_pca[0])

    train_dict = defaultdict(list)
    for index, cluster_label in enumerate(k_means.labels_):
        train_dict[f'{cluster_label}'].append(x_train_pca[index])

    for index, test_label in enumerate(predict_label):
        distance_array = np.array([])
        for train_data_point in train_dict[f'{test_label}']:
            distance = train_data_point - predict_label[index]
            norm = np.linalg.norm(distance)
            distance_array = np.append(distance_array, norm)
        print('====================================')
        print(f'Predict data {index} is in Group {test_label}')
        print('Best data point:', (distance_array.argmin(), np.min(distance_array)))
        print(label_data[distance_array.argmin()])
        st.subheader('The Best Restaurant for you')
        image_from_internet = ''
        if f'{label_data[distance_array.argmin()]}' == 'Shake Shack':
            image_from_internet = requests.get(
                'https://play-lh.googleusercontent.com/WQsoRg7epNpgJRrEMkkLJqheDekpJfvuDX5UFuk3Et67i5472dc92XfQu_hc1bIi6pI')
            # image = Image.open("D:\Research data\SSID\Advanced Computer Python\Python-tools\Shack Shack.png")
        elif f'{label_data[distance_array.argmin()]}' == 'Burger King':
            image_from_internet = requests.get(
                'https://logos-world.net/wp-content/uploads/2020/05/Burger-King-Logo.png')
            # image = Image.open("D:\Research data\SSID\Advanced Computer Python\Python-tools\Burger King.png")
        elif f'{label_data[distance_array.argmin()]}' == "McDonald's":
            image_from_internet = requests.get(
                'https://1000logos.net/wp-content/uploads/2017/03/McDonalds-logo.png')
            # image = Image.open("D:\Research data\SSID\Advanced Computer Python\Python-tools\McDonalds.png")
        image_converted = BytesIO(image_from_internet.content)
        image = Image.open(image_converted)
        st.image(image, caption=f'{label_data[distance_array.argmin()]}')

    st.subheader('Data Visualization with respect to Survived')
    # left_column, right_column = st.columns(2)
    # with left_column:
    #     'Numerical Plot'
    #     num_feat = st.selectbox('Select Numerical Feature', df.select_dtypes('number').columns)
    #     fig = px.histogram(df, x=num_feat, color='survived')
    #     st.plotly_chart(fig, use_container_width=True)
    # with right_column:
    #     'Categorical column'
    #     cat_feat = st.selectbox('Select Categorical Feature', df.select_dtypes(exclude='number').columns)
    #     fig = px.histogram(df, x=cat_feat, color='survived')
    #     st.plotly_chart(fig, use_container_width=True)

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



    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


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


def synthetic_dataset():
    data_size = DATA_SIZE
    random.seed(0)
    np.random.seed(0)
    selection_dictionary = defaultdict()
    age_list = ['11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '70+']
    selection_dictionary['Age'] = pd.Series(random.choice(age_list) for _ in range(data_size))

    gender_list = ['Male', 'Female', 'Third gender']
    selection_dictionary['Gender'] = pd.Series(random.choice(gender_list) for _ in range(data_size))

    goal_list = ['Date', 'Business', 'Friends', 'Family', 'Alone', 'Other']
    selection_dictionary['Goal'] = pd.Series(random.choice(goal_list) for _ in range(data_size))

    hours_list = ['Brunch', 'Lunch', 'Dinner', 'Anytime']
    selection_dictionary['Hours'] = pd.Series(random.choice(hours_list) for _ in range(data_size))

    distance_list = ['0-25', '26-50', '51-100', 'idc']
    selection_dictionary['Distance'] = pd.Series(random.choice(distance_list) for _ in range(data_size))

    restaurant_list = ['Burger King', 'Shake Shack', "McDonald's"]
    selection_dictionary['Restaurant'] = pd.Series(random.choice(restaurant_list) for _ in range(data_size))

    selection_dictionary['Extra column'] = pd.Series(list(np.random.randint(2, size=data_size)))

    return pd.DataFrame(selection_dictionary)


def data_preprocess(filename, mode='Train', training_dataframe=None):
    """
    :param filename: str, the filename to be read into pandas
    :param mode: str, indicating the mode we are using (either Train or Test)
    :param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
                          (You will only use this when mode == 'Test')
    :return: Tuple(data, labels), if the mode is 'Train'
             data, if the mode is 'Test'
    """
    # data = pd.read_csv(filename)	 # Read the file in a dataframe form
    data = filename
    print('Data Head')
    print(data.head(5))
    # column_names = row_data.head(0).columns if you need all the column names
    dataframe_format = pd.DataFrame()
    if mode == 'Train':
        column_names = ['Age', 'Gender', 'Goal', 'Hours', 'Distance', 'Restaurant']

        encoding_list = ['Age', 'Gender', 'Goal', 'Hours', 'Distance']  # List for one-hot encoding
        for feature in encoding_list:
            data = one_hot_encoding(data, feature)
        labels = data['Restaurant']  # Save labels
        data.pop('Restaurant')
        data.pop('Extra column')
        dataframe_format = data.drop(data.index[1:], axis=0).replace(1, 0)
        return data, labels, dataframe_format

    elif mode == 'Test':
        column_names = ['Age', 'Gender', 'Goal', 'Hours', 'Distance']
        column_dictionary = defaultdict(list)
        column_dictionary['Age'] = ['11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '70+']
        column_dictionary['Gender'] = ['Male', 'Female', 'Third gender']
        column_dictionary['Goal'] = ['Date', 'Business', 'Friends', 'Family', 'Alone', 'Other']
        column_dictionary['Hours'] = ['Brunch', 'Lunch', 'Dinner', 'Anytime']
        column_dictionary['Distance'] = ['0-25', '26-50', '51-100', 'idc']
        for column_index, column in enumerate(column_dictionary):
            for item_index, item in enumerate(column_dictionary[column]):
                # data.loc[data[column] == column_dictionary[column][item_index], column] = item_index
                training_dataframe.loc[data[column] == item, f'{column}_{item}'] = 1
        data = training_dataframe    # Store data using training dataframe to fit the feature dimension
        return data


def one_hot_encoding(data, feature):
    """
    :param data: DataFrame, key is the column name, value is its data
    :param feature: str, the column name of interest
    :return data: DataFrame, remove the feature column and add its one-hot encoding features
    """
    data = pd.get_dummies(data, columns=[feature])
    return data


def user_input_feature():
    # separation = st.sidebar.slider("Separation", 0.7, 2.0, 0.7885)
    st.sidebar.title('User Preference')
    age = st.sidebar.selectbox(
        'In what age group are you?',
        ('11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '70+'))
    sex = st.sidebar.selectbox(
        "What's your gender?",
        ('Male', 'Female', 'Third gender'))
    goal = st.sidebar.selectbox(
        "What's your purpose?",
        ('Date', 'Business', 'Friends', 'Family', 'Alone', 'Other'))
    hours = st.sidebar.selectbox(
        "What time?",
        ['Brunch', 'Lunch', 'Dinner', 'Anytime'])
    distance = st.sidebar.selectbox(
        "How far from you (miles)?",
        ('0-25', '26-50', '51-100', 'idc'))

    feature_dictionary = defaultdict()
    feature_dictionary['Age'] = pd.Series(age)
    feature_dictionary['Gender'] = pd.Series(sex)
    feature_dictionary['Goal'] = pd.Series(goal)
    feature_dictionary['Hours'] = pd.Series(hours)
    feature_dictionary['Distance'] = pd.Series(distance)

    return pd.DataFrame(feature_dictionary)


if __name__ == '__main__':
    main()