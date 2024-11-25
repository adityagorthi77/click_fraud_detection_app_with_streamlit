#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd

# Load the model from the file
with open('final_random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
balanced_data = pd.read_pickle('cleaned_data.pkl')

# In[11]:


import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Load the pre-trained RandomForestClassifier
with open('final_random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Your preprocessing steps here
    # For example, convert categorical variables to numerical, handle missing values, etc.
    data['click_time'] = pd.to_datetime(data['click_time'])
    data['hour'] = data['click_time'].dt.hour

    # Drop the original 'click_time' column
    data = data.drop('click_time', axis=1)

    return data

# Function to make predictions using the loaded model
def make_predictions(model, input_data):
    predictions = model.predict(input_data)
    return predictions

# Function to display visualizations
def display_visualizations(data):
    # Your visualization code here
    # For example, plot charts, histograms, etc.
    pass
st.set_page_config(
    page_title="Talking data Click fraud Web App",
    page_icon=None,
    layout="wide",  # You can change this layout as per your preference
    initial_sidebar_state="expanded",
)

# Add your company logo or any other branding elements here
st.image('https://www.businessprocessincubator.com/wp-content/uploads/thumbnails/thumbnail-83782.jpg', use_column_width=True)

# Add your company logo or any other branding elements here
# st.image('path_to_your_logo.png', use_column_width=True)
# Sidebar options
st.sidebar.image("//Users/saikrishnaadityagorthi/desktop/sujithgo_bahalulk_sgorthi_phase_3/data/Picture1.jpeg")
option = st.sidebar.selectbox(
    'Choose an option:',
    ['Home', 'Predictions', 'Visualizations']
)

# Home section
if option == 'Home':
   
    st.write('This is the home section. Choose an option from the sidebar.')

# Predictions section
elif option == 'Predictions':
    st.subheader('Model Predictions')
    st.sidebar.header('User Input Features')

    ip = st.sidebar.number_input('IP', min_value=0.0, max_value=1e10, value=5e9, step=1e7, format="%.0f")
    app = st.sidebar.number_input('App', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    device = st.sidebar.number_input('Device', min_value=0.0, step=1.0)
    os = st.sidebar.number_input('OS', min_value=0.0, step=1.0)
    channel = st.sidebar.number_input('Channel', min_value=0.0, step=1.0)
    click_time = st.sidebar.date_input('Click Time')

    if st.sidebar.button('Validate'):
        user_input = pd.DataFrame({
            'ip': [ip],
            'app': [app],
            'device': [device],
            'os': [os],
            'channel': [channel],
            'click_time': [click_time]
        })

        processed_input = preprocess_input(user_input)
        predictions = make_predictions(model, processed_input)

        
        if predictions[0] == 1:
            st.write("The app is downloaded.")
            st.markdown('<span style="color:red; font-size:30px;">&#9888;</span>', unsafe_allow_html=True)
        else:
            st.write("The app is not downloaded.")

# Visualizations section
elif option == 'Visualizations':
    st.subheader('Data Visualizations')
    visualization_option = st.sidebar.selectbox(
        'Choose a visualization:',
        ['Top 30 Apps', 'Attributed Clicks per Hour','Top 30 Apps with Attribution', 'Top 30 Devices with Attribution', 'Top 30 OS with Attribution', 'Top 30 Channels', 'Top 30 Attributed Channels', 'Top 50 IPs', 'Top 50 Attributed IPs']
    )
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Update balanced_data with user-uploaded data
        balanced_data = pd.read_csv(uploaded_file)
    else:
        # Display a message or default visualizations if no file is uploaded
        st.warning("No file uploaded. Using default data.")

    if visualization_option == 'Top 30 Apps':
        st.subheader('Top 30 Apps in Balanced Data')
        # Get the top 30 values for the 'app' column
        top_apps = balanced_data['app'].value_counts().nlargest(30)

        # Set the figure size
        fig, ax = plt.subplots(figsize=(12, 6))

        # Use Seaborn for better visualization
        sns.barplot(x=top_apps.index, y=top_apps, palette='viridis', ax=ax)

        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add labels and title
        ax.set_xlabel('App')
        ax.set_ylabel('Count')
        ax.set_title('Top 30 Apps in Balanced Data')

        # Use st.pyplot() with the Matplotlib figure explicitly
        st.pyplot(fig)
    if visualization_option == 'Attributed Clicks per Hour':
        st.subheader('Attributed Clicks per Hour')
        # Convert 'click_time' to datetime
        balanced_data['click_time'] = pd.to_datetime(balanced_data['click_time'])

        # Extract the hour from 'click_time'
        balanced_data['hour'] = balanced_data['click_time'].dt.hour

        # Get the unique hours and sort them
        sorted_hours = sorted(balanced_data['hour'].unique())

        # Plotting for attributed clicks per hour
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='hour', data=balanced_data, palette='muted', order=sorted_hours, ax=ax)
        ax.set_title('Attributed Clicks per Hour')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Count')
        ax.tick_params(rotation=0)  # Rotate x-axis labels if needed

        # Use st.pyplot() with the Matplotlib figure explicitly
        st.pyplot(fig)
    if visualization_option == 'Top 30 Apps with Attribution':
        st.subheader('Top 30 Apps with Attribution')

        # Filter data where is_attributed is equal to 1
        attributed_data = balanced_data[balanced_data['is_attributed'] == 1]

        # Get the top 30 apps with attribution
        top_apps_attributed = attributed_data['app'].value_counts().nlargest(30)

        # Set the figure size
        fig, ax = plt.subplots(figsize=(14, 7))

        # Use Seaborn for better visualization
        sns.barplot(x=top_apps_attributed.index, y=top_apps_attributed, palette='viridis', ax=ax)

        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add labels and title
        ax.set_xlabel('App')
        ax.set_ylabel('Count')
        ax.set_title('Top 30 Apps with Attribution')

        # Use st.pyplot() with the Matplotlib figure explicitly
        st.pyplot(fig)
    if visualization_option == 'Top 30 Devices with Attribution':
        st.subheader('Top 30 Devices with Attribution')

        # Filter data where is_attributed is equal to 1
        attributed_data = balanced_data[balanced_data['is_attributed'] == 1]

        # Get the top 30 devices with attribution
        top_devices_attributed = attributed_data['device'].value_counts().nlargest(30)

        # Set the figure size
        fig, ax = plt.subplots(figsize=(12, 6))

        # Use Seaborn for better visualization
        sns.barplot(x=top_devices_attributed.index, y=top_devices_attributed, ax=ax)

        # Add labels and title
        ax.set_title('Top 30 Devices with Attribution')
        ax.set_xlabel('Device')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation_mode='anchor', ha='right')

        # Use st.pyplot() with the Matplotlib figure explicitly
        st.pyplot(fig)
    if visualization_option == 'Top 30 OS with Attribution':
        st.subheader('Top 30 OS with Attribution')

        # Filter data where is_attributed is equal to 1
        attributed_data = balanced_data[balanced_data['is_attributed'] == 1]

        # Get the top 30 OS with attribution
        top_os_attributed = attributed_data['os'].value_counts().nlargest(30)

        # Set the figure size
        fig, ax = plt.subplots(figsize=(12, 6))

        # Use Seaborn for better visualization
        sns.barplot(x=top_os_attributed.index, y=top_os_attributed, ax=ax)

        # Add labels and title
        ax.set_title('Top 30 OS with Attribution')
        ax.set_xlabel('OS')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation_mode='anchor', ha='right')

        # Use st.pyplot() with the Matplotlib figure explicitly
        st.pyplot(fig)
    if visualization_option == 'Top 30 Channels':
        st.subheader('Top 30 Channels')

        # Calculate the number of clicks per channel
        clicks_per_channel = balanced_data['channel'].value_counts()

        # Get the top 30 channels
        top_channels = clicks_per_channel.nlargest(30)

        # Set the figure size
        fig, ax = plt.subplots(figsize=(14, 7))

        # Use Seaborn for better visualization
        sns.barplot(x=top_channels.index, y=top_channels, palette='viridis', ax=ax)

        # Add labels and title
        ax.set_title('Top 30 Channels')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation_mode='anchor', ha='right')

        # Use st.pyplot() with the Matplotlib figure explicitly
        st.pyplot(fig)
    if visualization_option == 'Top 30 Attributed Channels':
        st.subheader('Top 30 Attributed Channels')

        # Filter data where is_attributed is equal to 1
        attributed_data = balanced_data[balanced_data['is_attributed'] == 1]

        # Calculate the number of attributed clicks per channel
        attributed_clicks_per_channel = attributed_data['channel'].value_counts()

        # Get the top 30 attributed channels
        top_attributed_channels = attributed_clicks_per_channel.nlargest(30)

        # Set the figure size
        fig, ax = plt.subplots(figsize=(14, 7))

        # Use Seaborn for better visualization
        sns.barplot(x=top_attributed_channels.index, y=top_attributed_channels, palette='mako', ax=ax)

        # Add labels and title
        ax.set_title('Top 30 Attributed Channels')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation_mode='anchor', ha='right')
        # Use st.pyplot() with the Matplotlib figure explicitly
        st.pyplot(fig)
    if visualization_option == 'Top 50 IPs':
        st.subheader('Top 50 IPs')

        # Calculate the number of clicks per IP
        clicks_per_ip = balanced_data['ip'].value_counts()

        # Get the top 50 IPs
        top_ips = clicks_per_ip.nlargest(50)

        # Set the figure size
        fig, ax = plt.subplots(figsize=(14, 7))

        # Use Seaborn for better visualization
        sns.barplot(x=top_ips.index, y=top_ips, palette='viridis', ax=ax)

        # Add labels and title
        ax.set_title('Top 50 IPs')
        ax.set_xlabel('IP')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation_mode='anchor', ha='right')

        # Use st.pyplot() with the Matplotlib figure explicitly
        st.pyplot(fig)
    if visualization_option == 'Top 50 Attributed IPs':
        st.subheader('Top 50 Attributed IPs')

        # Filter data where is_attributed is equal to 1
        attributed_data = balanced_data[balanced_data['is_attributed'] == 1]

        # Calculate the number of attributed clicks per IP
        attributed_clicks_per_ip = attributed_data['ip'].value_counts()

        # Get the top 50 attributed IPs
        top_attributed_ips = attributed_clicks_per_ip.nlargest(50)

        # Set the figure size
        fig, ax = plt.subplots(figsize=(14, 7))

        # Use Seaborn for better visualization
        sns.barplot(x=top_attributed_ips.index, y=top_attributed_ips, palette='mako', ax=ax)

        # Add labels and title
        ax.set_title('Top 50 Attributed IPs')
        ax.set_xlabel('IP')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation_mode='anchor', ha='right')

        # Use st.pyplot() with the Matplotlib figure explicitly
        st.pyplot(fig)

    # Add more visualizations as needed...

# You can add more sections/options as needed
