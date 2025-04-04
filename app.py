import streamlit as st
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import smtplib
from email.mime.text import MIMEText
import time
import os

# Load the trained model and autoencoder
model = load_model("rul_lstm_model.h5")
autoencoder = load_model("autoencoder.keras")
features = ['T50', 'P24', 'P15', 'P21', 'P40', 'Ps30', 'T24', 'Wf', 'P50', 'T30', 'T48', 'Nc', 'Nf']
sequence_length = 50

# Load the dataset for EDA
data_path = '/content/CMAPSSv2/data_set/N-CMAPSS_DS01-005.h5'
if not os.path.exists(data_path):
    st.error(f"Dataset file not found at {data_path}. Please ensure the N-CMAPSS_DS01-005.h5 file is in the correct directory.")
    st.stop()

# Load and preprocess the data from the HDF5 file
with h5py.File(data_path, 'r') as f:
    # Load all units (not just unit 1)
    a_data = f['A_dev'][:]
    x_s_data = f['X_s_dev'][:]
    w_data = f['W_dev'][:]
    y_data = f['Y_dev'][:]

    # Column names
    columns = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf',
               'alt', 'Mach', 'TRA', 'T2', 'unit', 'cycle', 'Fc', 'hs', 'rul']

    # Combine into DataFrame
    df = pd.DataFrame(x_s_data, columns=columns[:14])  # X_s
    df[columns[14:18]] = w_data  # W
    df[columns[18:22]] = a_data  # A
    df['rul'] = y_data

    # Fix time_step
    df['time_step'] = np.arange(len(df))

    # Recalculate RUL
    max_time = df['time_step'].max()
    df['rul_calculated'] = max_time - df['time_step']

# Filter for the first 10 cycles (flights)
df_larger = df[df['cycle'] <= 10].copy()

# Function to create sequences
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
    return np.array(X)

# Function to send email alerts
def send_alert(rul, threshold=10):
    if rul < threshold:
        msg = MIMEText(f"Alert: RUL dropped to {rul:.2f} flights!")
        msg['Subject'] = 'RUL Alert'
        msg['From'] = 'your_email@example.com'
        msg['To'] = 'maintenance_team@example.com'
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your_email@example.com', 'your_password')
            server.sendmail('your_email@example.com', 'maintenance_team@example.com', msg.as_string())

# Streamlit app
st.title("Comprehensive IoT Predictive Maintenance Dashboard")
st.header("Predict Remaining Useful Life (RUL) for Aircraft Engine")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["RUL Prediction", "Real-Time Streaming", "Exploratory Data Analysis", "Explainable AI", "Anomaly Detection"])

if page == "RUL Prediction":
    # Model performance
    st.subheader("Model Performance")
    st.write("Test RMSE: 2.95 flights (predictions within ±3 flights)")
    st.write(f"RUL Variance (First 10 Flights): {df_larger['rul'].var():.4f}")

    # Unit selection
    units = df_larger['unit'].unique()
    selected_unit = st.selectbox("Select Engine Unit (First 10 Flights)", units)
    df_unit = df_larger[df_larger['unit'] == selected_unit]

    # File upload for input data
    st.subheader("Upload Sensor Data (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file with sensor data", type="csv")

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(df_input.head())

        if all(feature in df_input.columns for feature in features):
            scaler = MinMaxScaler()
            df_input[features] = scaler.fit_transform(df_input[features])

            data_input = df_input[features].values
            if len(data_input) >= sequence_length:
                X_input = create_sequences(data_input, sequence_length)
                X_input = np.concatenate([X_input, np.zeros((X_input.shape[0], X_input.shape[1], 1))], axis=-1)

                # Predict RUL
                y_pred = model.predict(X_input)
                st.subheader("Predicted RUL")
                st.write(f"Predicted RUL: {y_pred[-1][0]:.2f} flights")

                # Send alert if RUL is low
                send_alert(y_pred[-1][0])

                # Sensor trends
                st.subheader("Sensor Trends (Last 50 Time Steps)")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_input['T30'].tail(50), label='T30 (HPC Outlet Temp)', color='blue')
                ax.plot(df_input['T50'].tail(50), label='T50', color='green')
                ax.plot(df_input['P24'].tail(50), label='P24', color='red')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Sensor Value (Normalized)')
                ax.legend()
                st.pyplot(fig)
            else:
                st.error(f"Input data must have at least {sequence_length} rows.")
        else:
            st.error("Uploaded CSV must contain the following columns: " + ", ".join(features))
    else:
        st.info("Please upload a CSV file to predict RUL.")

elif page == "Real-Time Streaming":
    st.subheader("Real-Time RUL Prediction")

    # Placeholder for real-time data
    if 'data_buffer' not in st.session_state:
        st.session_state.data_buffer = pd.DataFrame(columns=features)

    # Simulate real-time data
    def simulate_new_data():
        new_row = np.random.uniform(0, 1, len(features))
        return pd.DataFrame([new_row], columns=features)

    placeholder = st.empty()
    while True:
        new_data = simulate_new_data()
        st.session_state.data_buffer = pd.concat([st.session_state.data_buffer, new_data], ignore_index=True)

        if len(st.session_state.data_buffer) > sequence_length:
            st.session_state.data_buffer = st.session_state.data_buffer.tail(sequence_length)

        if len(st.session_state.data_buffer) == sequence_length:
            data_input = st.session_state.data_buffer[features].values
            X_input = np.array([data_input])
            X_input = np.concatenate([X_input, np.zeros((X_input.shape[0], X_input.shape[1], 1))], axis=-1)
            y_pred = model.predict(X_input)
            placeholder.write(f"Predicted RUL: {y_pred[0][0]:.2f} flights")

            # Send alert if RUL is low
            send_alert(y_pred[0][0])

            # Plot sensor trends
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(st.session_state.data_buffer['T30'], label='T30 (HPC Outlet Temp)', color='blue')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('T30 (Normalized)')
            ax.legend()
            st.pyplot(fig)

        time.sleep(1)

elif page == "Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")

    # RUL Distribution
    st.write("### RUL Distribution Across Units (First 10 Flights)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='unit', y='rul', data=df_larger, ax=ax)
    ax.set_title("RUL Distribution Across Units (First 10 Flights)")
    st.pyplot(fig)

    # Sensor Trends
    st.write("### Sensor Trends Over Time (Unit 1, First 10 Flights)")
    unit_1 = df_larger[df_larger['unit'] == 1]
    sensors = ['T50', 'P24', 'T30', 'T48']
    fig, ax = plt.subplots(figsize=(10, 6))
    for sensor in sensors:
        ax.plot(unit_1['time_step'], unit_1[sensor], label=sensor)
    ax.set_title("Sensor Trends Over Time (Unit 1, First 10 Flights)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Sensor Value")
    ax.legend()
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("### Correlation Heatmap (First 10 Flights)")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df_larger[features + ['rul']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Correlation Heatmap of Sensors and RUL (First 10 Flights)")
    st.pyplot(fig)

elif page == "Explainable AI":
    st.subheader("Explainable AI (SHAP Analysis)")

    # Prepare data for SHAP
    data_larger = df_larger[features].values
    X_larger = create_sequences(data_larger, sequence_length)
    X_explain = X_larger[:100]  # Use 100 sequences for explanation

    explainer = shap.DeepExplainer(model, X_larger[:100])
    shap_values = explainer.shap_values(X_explain)

    # SHAP summary plot
    st.write("### SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features=features, plot_type="bar")
    st.pyplot(fig)

elif page == "Anomaly Detection":
    st.subheader("Anomaly Detection")

    # File upload for anomaly detection
    uploaded_file = st.file_uploader("Upload Sensor Data for Anomaly Detection (CSV)", type="csv")
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        if all(feature in df_input.columns for feature in features):
            scaler = MinMaxScaler()
            df_input[features] = scaler.fit_transform(df_input[features])
            data_input = df_input[features].values
            if len(data_input) >= sequence_length:
                X_input = create_sequences(data_input, sequence_length)
                reconstructions = autoencoder.predict(X_input)
                mse = np.mean(np.square(X_input - reconstructions), axis=(1, 2))
                threshold = np.percentile(mse, 95)
                anomalies = mse > threshold
                st.write("Anomalies Detected at Indices:", np.where(anomalies)[0])
            else:
                st.error(f"Input data must have at least {sequence_length} rows.")
        else:
            st.error("Uploaded CSV must contain the following columns: " + ", ".join(features))
    else:
        st.info("Please upload a CSV file for anomaly detection.")

