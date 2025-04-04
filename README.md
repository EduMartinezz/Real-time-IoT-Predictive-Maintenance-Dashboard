## Comprehensive IoT Predictive Maintenance Dashboard for Aircraft Engines

## Project Overview
This project implements a predictive maintenance dashboard for aircraft engines using the N-CMAPSS dataset. The dashboard provides Remaining Useful Life (RUL) predictions, anomaly detection, exploratory data analysis (EDA), and explainable AI (SHAP analysis) using a combination of LSTM models and autoencoders. The app is built with Streamlit and can be run locally or deployed on the cloud.
The project was developed in Google Colab, with optimizations to ensure it runs on the free tier despite the large dataset size (14.7 GB). Key features include RUL prediction, real-time monitoring, and anomaly detection, making it a comprehensive tool for aircraft engine maintenance.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Cloud Deployment](#cloud-deployment)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features
- **RUL Prediction**: Upload sensor data (CSV) to predict the RUL of an engine using an LSTM model.
- **Real-Time Monitoring**: Simulate real-time RUL predictions with streaming data.
- **Exploratory Data Analysis (EDA)**: Visualize RUL distributions, sensor trends, and correlations for the first 10 flights of a selected unit.
- **Explainable AI (SHAP)**: Provides feature importance for RUL predictions using SHAP analysis.
- **Anomaly Detection**: Detect anomalies in sensor data using an autoencoder.
- **Email Alerts**: Send email alerts when RUL drops below a threshold (requires email configuration).

## Dataset
The project uses the [N-CMAPSS dataset](https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip) (Turbofan Engine Degradation Simulation Data Set 2), which contains run-to-failure data for aircraft engines under real flight conditions. The dataset includes:
- Sensor measurements (e.g., temperatures, pressures, fan speeds).
- Scenario descriptors (e.g., altitude, Mach number).
- Auxiliary data (e.g., unit number, cycle).
- Target variable (RUL).

**Size**: The dataset is 14.7 GB when extracted. Due to GitHub’s file size limits, it is not included in this repository. The code automatically downloads and extracts the dataset during setup.

## Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`:

streamlit==1.24.0 
numpy==1.24.3 
pandas==2.0.2 
h5py==3.9.0 
tensorflow==2.12.0 
scikit-learn==1.2.2 
matplotlib==3.7.1 
seaborn==0.12.2 
shap==0.42.0 
smtplib 
pyngrok==7.0.0


## Setup Instructions
**Clone the Repository**:
 git clone https://github.com/your-username/your-repo-name.git
 cd your-repo-name

**Create a Virtual Environment (optional )**:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

**Install Dependencies**:
pip install -r requirements.txt

**Download the Dataset**:
- The dataset is automatically downloaded and extracted when you run the code (Step 1). Alternatively, you can manually download it from here and place it in the project directory as CMAPSSv2.zip.

**Run the Project**:
- The project is divided into 9 steps, each in a separate Jupyter notebook cell or script. You can run the steps sequentially in a Jupyter notebook or as a single script (main.py).
- To run the dashboard locally:
streamlit run app.py
To deploy using ngrok (for public access), set your ngrok authtoken in app.py and run the deployment step (Step 9).

## Usage
1. Run the Dashboard Locally:
- After setting up the project, run:
  streamlit run app.py
  Open the provided URL (e.g., http://localhost:8501) in your browser.

2. Navigate the Dashboard:
• RUL Prediction: Upload a CSV file with sensor data (e.g., test_sensor_data.csv) to predict RUL.
• Real-Time Streaming: View simulated real-time RUL predictions.
• Exploratory Data Analysis: Visualize RUL distributions, sensor trends, and correlations.
• Explainable AI: View SHAP analysis for feature importance.
• Anomaly Detection: Upload a CSV file to detect anomalies in sensor data.

3.Example Input:
• Use test_sensor_data.csv (generated in Step 8) as a sample input for RUL prediction and anomaly detection. The file contains the last 100 rows of normalized sensor data for the first 10 flights.

Cloud Deployment(Optional)
Option 1: Streamlit Community Cloud (Recommended for Prototyping)
1.	Push your repository to GitHub.
2.	Sign up for Streamlit Community Cloud (https://streamlit.io/cloud).
3.	Connect your GitHub account and select your repository.
4.	Specify the main script (app.py) and deploy the app.
5.	Note: The N-CMAPSS dataset is too large for Streamlit Cloud’s storage. Modify app.py to download the dataset from a cloud storage service (e.g., Google Drive, AWS S3) at runtime:

import gdown
url = "your-google-drive-link-to-CMAPSSv2.zip"
gdown.download(url, "CMAPSSv2.zip", quiet=False)

## Project Structure
your-repo-name/
│
├── app.py                  # Streamlit dashboard script
├── main.py                 # Main script with all steps (optional)
├── requirements.txt        # Python dependencies
├── test_sensor_data.csv    # Sample input data for the dashboard
├── rul_lstm_model.h5       # Trained LSTM model for RUL prediction
├── autoencoder.keras       # Trained autoencoder for anomaly detection
├── CMAPSSv2/               # Directory for the N-CMAPSS dataset (created during setup)
│   ├── train_subset.csv
│   ├── train_flight_subset.csv
│   └── data_set/
│       └── N-CMAPSS_DS01-005.h5
└── README.md               # This file

Results
• RUL Prediction: The LSTM model achieves a test RMSE of 2.95 flights, meaning predictions are within ±3 flights of the actual RUL.
• Anomaly Detection: The autoencoder detects anomalies in the test set, identifying potential issues in sensor data.
• EDA: Visualizations show RUL distributions, sensor trends, and correlations for the first 10 flights.
• SHAP Analysis: SHAP plots highlight the most important sensor features for RUL predictions.

**Sample Output**:
• Predicted RUL for test_sensor_data.csv: 38.95 flights.
• RUL Variance (First 10 Flights): 98.1031.
• Sensor Trends: Sharp drop in sensor values (T30, T50, P24) in the last 10 time steps, indicating potential degradation.

Contributing
Contributions are welcome! Please follow these steps:
- Fork the repository.
- Create a new branch: git checkout -b feature/your-feature-name.
- Make your changes and commit: git commit -m "Add your feature".
- Push to your branch: git push origin feature/your-feature-name.
- Create a pull request.

License
This project is licensed under the MIT License. 


### Additional Notes
1. **Improving RUL Predictions**:
   - The predicted RUL (38.95 flights) is lower than expected (90–99). To improve predictions, consider:
     - Retraining the model on a larger dataset (e.g., all units and cycles) to improve generalization.
     - Adding more features (e.g., scenario descriptors like `alt`, `Mach`) to the model.
     - Tuning the LSTM architecture (e.g., more layers, different units) or hyperparameters (e.g., learning rate, batch size).

2. **Enhancing the Dashboard**:
   - Add a feature to display the actual RUL alongside the predicted RUL for comparison.
   - Include a download button for EDA plots and SHAP analysis results.
   - Add a page to visualize the autoencoder’s reconstruction errors for anomaly detection.

3. **README Customization**:
   - Replace `your-username` and `your-repo-name` in the README with your actual GitHub username and repository name.
   - If you have a specific license (e.g., MIT, Apache), create a `LICENSE` file and link to it in the README.
   - Add screenshots of the dashboard (e.g., the sensor trends plot) to the README to make it more visually appealing.

Let me know if you’d like to explore any of these improvements or need help with deployment!



