```python
#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Talking data Click fraud Web App",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Adityagorthi77/click-fraud-random-forest-model",
        filename="final_random_forest_model.pkl"
    )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


# ============================================================
# LOAD VISUALIZATION DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_pickle("cleaned_data.pkl")


# Load model and data
try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the fraud detection model from Hugging Face.")
    st.exception(e)
    st.stop()


try:
    balanced_data = load_data()
except Exception as e:
    st.error("Unable to load cleaned_data.pkl.")
    st.exception(e)
    st.stop()


# ============================================================
# PREPROCESS INPUT DATA
# ============================================================

def preprocess_input(data):
    data = data.copy()

    data["click_time"] = pd.to_datetime(data["click_time"])
    data["hour"] = data["click_time"].dt.hour

    # Drop original click_time column
    data = data.drop("click_time", axis=1)

    return data


# ============================================================
# MAKE PREDICTIONS
# ============================================================

def make_predictions(model, input_data):
    predictions = model.predict(input_data)
    return predictions


# ============================================================
# HOME PAGE
# ============================================================

st.image(
    "https://www.businessprocessincubator.com/wp-content/uploads/thumbnails/thumbnail-83782.jpg",
    use_container_width=True
)

st.sidebar.image(
    "images.jpeg",
    use_container_width=True
)


option = st.sidebar.selectbox(
    "Choose an option:",
    ["Home", "Predictions", "Visualizations"]
)


# ============================================================
# HOME SECTION
# ============================================================

if option == "Home":

    st.title("Click Fraud Detection Web App")

    st.write(
        "This application uses a machine learning model to predict "
        "whether an advertising click is likely to result in an app download."
    )

    st.info(
        "Use the sidebar to navigate to Predictions or Visualizations."
    )


# ============================================================
# PREDICTIONS SECTION
# ============================================================

elif option == "Predictions":

    st.subheader("Model Predictions")

    st.sidebar.header("User Input Features")

    ip = st.sidebar.number_input(
        "IP",
        min_value=0.0,
        max_value=1e10,
        value=5e9,
        step=1e7,
        format="%.0f"
    )

    app = st.sidebar.number_input(
        "App",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

    device = st.sidebar.number_input(
        "Device",
        min_value=0.0,
        step=1.0
    )

    os = st.sidebar.number_input(
        "OS",
        min_value=0.0,
        step=1.0
    )

    channel = st.sidebar.number_input(
        "Channel",
        min_value=0.0,
        step=1.0
    )

    click_time = st.sidebar.date_input(
        "Click Time"
    )

    if st.sidebar.button("Validate"):

        user_input = pd.DataFrame({
            "ip": [ip],
            "app": [app],
            "device": [device],
            "os": [os],
            "channel": [channel],
            "click_time": [click_time]
        })

        try:

            processed_input = preprocess_input(user_input)

            predictions = make_predictions(
                model,
                processed_input
            )

            if predictions[0] == 1:

                st.success("The app is downloaded.")

                st.markdown(
                    '<span style="color:red; font-size:30px;">&#9888;</span>',
                    unsafe_allow_html=True
                )

            else:

                st.info("The app is not downloaded.")

        except Exception as e:

            st.error("An error occurred while making the prediction.")
            st.exception(e)


# ============================================================
# VISUALIZATIONS SECTION
# ============================================================

elif option == "Visualizations":

    st.subheader("Data Visualizations")

    visualization_option = st.sidebar.selectbox(
        "Choose a visualization:",
        [
            "Top 30 Apps",
            "Attributed Clicks per Hour",
            "Top 30 Apps with Attribution",
            "Top 30 Devices with Attribution",
            "Top 30 OS with Attribution",
            "Top 30 Channels",
            "Top 30 Attributed Channels",
            "Top 50 IPs",
            "Top 50 Attributed IPs"
        ]
    )

    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            visualization_data = pd.read_csv(uploaded_file)

            st.success("CSV file uploaded successfully.")

        except Exception as e:

            st.error("Unable to read the uploaded CSV file.")
            st.exception(e)
            st.stop()

    else:

        visualization_data = balanced_data

        st.warning(
            "No file uploaded. Using default data."
        )


    # ========================================================
    # TOP 30 APPS
    # ========================================================

    if visualization_option == "Top 30 Apps":

        st.subheader("Top 30 Apps in Balanced Data")

        top_apps = (
            visualization_data["app"]
            .value_counts()
            .nlargest(30)
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            x=top_apps.index,
            y=top_apps.values,
            ax=ax
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

        ax.set_xlabel("App")
        ax.set_ylabel("Count")
        ax.set_title("Top 30 Apps in Balanced Data")

        st.pyplot(fig)


    # ========================================================
    # ATTRIBUTED CLICKS PER HOUR
    # ========================================================

    elif visualization_option == "Attributed Clicks per Hour":

        st.subheader("Attributed Clicks per Hour")

        visualization_data = visualization_data.copy()

        visualization_data["click_time"] = pd.to_datetime(
            visualization_data["click_time"]
        )

        visualization_data["hour"] = (
            visualization_data["click_time"].dt.hour
        )

        sorted_hours = sorted(
            visualization_data["hour"].dropna().unique()
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.countplot(
            x="hour",
            data=visualization_data,
            order=sorted_hours,
            ax=ax
        )

        ax.set_title("Attributed Clicks per Hour")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")

        st.pyplot(fig)


    # ========================================================
    # TOP 30 APPS WITH ATTRIBUTION
    # ========================================================

    elif visualization_option == "Top 30 Apps with Attribution":

        st.subheader("Top 30 Apps with Attribution")

        attributed_data = visualization_data[
            visualization_data["is_attributed"] == 1
        ]

        top_apps_attributed = (
            attributed_data["app"]
            .value_counts()
            .nlargest(30)
        )

        fig, ax = plt.subplots(figsize=(14, 7))

        sns.barplot(
            x=top_apps_attributed.index,
            y=top_apps_attributed.values,
            ax=ax
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

        ax.set_xlabel("App")
        ax.set_ylabel("Count")
        ax.set_title("Top 30 Apps with Attribution")

        st.pyplot(fig)


    # ========================================================
    # TOP 30 DEVICES WITH ATTRIBUTION
    # ========================================================

    elif visualization_option == "Top 30 Devices with Attribution":

        st.subheader("Top 30 Devices with Attribution")

        attributed_data = visualization_data[
            visualization_data["is_attributed"] == 1
        ]

        top_devices_attributed = (
            attributed_data["device"]
            .value_counts()
            .nlargest(30)
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            x=top_devices_attributed.index,
            y=top_devices_attributed.values,
            ax=ax
        )

        ax.set_title("Top 30 Devices with Attribution")
        ax.set_xlabel("Device")
        ax.set_ylabel("Count")

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

        st.pyplot(fig)


    # ========================================================
    # TOP 30 OS WITH ATTRIBUTION
    # ========================================================

    elif visualization_option == "Top 30 OS with Attribution":

        st.subheader("Top 30 OS with Attribution")

        attributed_data = visualization_data[
            visualization_data["is_attributed"] == 1
        ]

        top_os_attributed = (
            attributed_data["os"]
            .value_counts()
            .nlargest(30)
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            x=top_os_attributed.index,
            y=top_os_attributed.values,
            ax=ax
        )

        ax.set_title("Top 30 OS with Attribution")
        ax.set_xlabel("OS")
        ax.set_ylabel("Count")

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

        st.pyplot(fig)


    # ========================================================
    # TOP 30 CHANNELS
    # ========================================================

    elif visualization_option == "Top 30 Channels":

        st.subheader("Top 30 Channels")

        clicks_per_channel = (
            visualization_data["channel"]
            .value_counts()
        )

        top_channels = clicks_per_channel.nlargest(30)

        fig, ax = plt.subplots(figsize=(14, 7))

        sns.barplot(
            x=top_channels.index,
            y=top_channels.values,
            ax=ax
        )

        ax.set_title("Top 30 Channels")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Count")

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

        st.pyplot(fig)


    # ========================================================
    # TOP 30 ATTRIBUTED CHANNELS
    # ========================================================

    elif visualization_option == "Top 30 Attributed Channels":

        st.subheader("Top 30 Attributed Channels")

        attributed_data = visualization_data[
            visualization_data["is_attributed"] == 1
        ]

        attributed_clicks_per_channel = (
            attributed_data["channel"]
            .value_counts()
        )

        top_attributed_channels = (
            attributed_clicks_per_channel
            .nlargest(30)
        )

        fig, ax = plt.subplots(figsize=(14, 7))

        sns.barplot(
            x=top_attributed_channels.index,
            y=top_attributed_channels.values,
            ax=ax
        )

        ax.set_title("Top 30 Attributed Channels")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Count")

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

        st.pyplot(fig)


    # ========================================================
    # TOP 50 IPS
    # ========================================================

    elif visualization_option == "Top 50 IPs":

        st.subheader("Top 50 IPs")

        clicks_per_ip = (
            visualization_data["ip"]
            .value_counts()
        )

        top_ips = clicks_per_ip.nlargest(50)

        fig, ax = plt.subplots(figsize=(14, 7))

        sns.barplot(
            x=top_ips.index,
            y=top_ips.values,
            ax=ax
        )

        ax.set_title("Top 50 IPs")
        ax.set_xlabel("IP")
        ax.set_ylabel("Count")

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

        st.pyplot(fig)


    # ========================================================
    # TOP 50 ATTRIBUTED IPS
    # ========================================================

    elif visualization_option == "Top 50 Attributed IPs":

        st.subheader("Top 50 Attributed IPs")

        attributed_data = visualization_data[
            visualization_data["is_attributed"] == 1
        ]

        attributed_clicks_per_ip = (
            attributed_data["ip"]
            .value_counts()
        )

        top_attributed_ips = (
            attributed_clicks_per_ip
            .nlargest(50)
        )

        fig, ax = plt.subplots(figsize=(14, 7))

        sns.barplot(
            x=top_attributed_ips.index,
            y=top_attributed_ips.values,
            ax=ax
        )

        ax.set_title("Top 50 Attributed IPs")
        ax.set_xlabel("IP")
        ax.set_ylabel("Count")

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

        st.pyplot(fig)
```
