# 🔧 Real-Time IoT Predictive Maintenance Dashboard

A **real-time machine learning web app** that simulates IoT sensor streams and predicts equipment failure risk.  
This project demonstrates a **full end-to-end workflow**: data streaming, feature engineering, anomaly detection, online learning, drift monitoring, and deployment via Streamlit.

---

## 📖 Project Overview
- Simulates **industrial IoT sensor data** (vibration, temperature, pressure, RPM).  
- Maintains a **sliding time window** of live signals for monitoring.  
- Computes **statistical + anomaly features** (z-scores, EWMA, rolling patterns).  
- Blends **unsupervised anomaly scores** with **online learning risk scores**.  
- Provides an interactive **Streamlit dashboard** with live plots, alerts, and drift checks.  

---

## 🚀 Features

### 1. Data Streaming
- Synthetic simulator generates live signals for multiple sensors.  
- Supports **drift injection** (to test stability) and **failure spikes** (to trigger alerts).  
- Rolling window ensures only the most recent data is kept.  

### 2. Feature Engineering
- **Robust z-scores** (Median Absolute Deviation).  
- **EWMA (Exponentially Weighted Moving Average)**.  
- Real-time derived anomaly features per sensor.  

### 3. Modeling
- **Online classifier**: `SGDClassifier` (log-loss) with `partial_fit` updates.  
- Trains incrementally when weak labels are available.  
- Blends anomaly detection with supervised learning via a weighted risk score.  

### 4. Interactive Dashboard
- **Dual-axis plots**: vibration (blue) + temperature (orange).  
- **Risk plots** with configurable threshold.  
- **Sidebar controls**:
  - Alert threshold  
  - Anomaly/model blending  
  - Online training rate  
  - Window size (plot last N points)  
- **Sidebar KPIs**:
  - Active alerts count  
  - Average + max risk across sensors  
  - Per-sensor risk values  

### 5. Alerts
- Alerts triggered when risk ≥ threshold.  
- Cooldown prevents spamming.  
- Alerts displayed with timestamp, risk, and features.  

### 6. Drift & Data Quality
- **Population Stability Index (PSI)**: feature drift vs baseline.  
- Data quality checks: missing values, flatlines, excessive spikes.  
- Drift levels:
  - PSI < 0.10 = stable  
  - 0.10–0.25 = moderate shift  
  - >0.25 = significant drift  

---

## 📊 Example Dashboard

**Controls + KPIs + Live Monitor**

![Controls and KPIs](screenshots/controls_kpis.png)

**Dual-axis vibration/temperature & Risk plots**

![Sensor plots](screenshots/sensor_plots.png)

**Alerts + Drift checks**

![Alerts and Drift](screenshots/alerts_drift.png)

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`  
- **Deployment:** Streamlit + ngrok (for Colab public URL)  
- **Environment:** Google Colab / Local Python  

---

## ▶️ Quickstart

### 1. Clone repo
git clone https://github.com/your-username/iot-predictive-maintenance.git
cd iot-predictive-maintenance

### 2. Install requirements
pip install -r requirements.txt

### 3. Run app (local)
streamlit run rt_maint.py

### 4. Run app (Colab with ngrok)
!pip install streamlit pyngrok
!pkill -f streamlit || true
!streamlit run rt_maint.py --server.port 8501 &>/content/logs.txt &
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
print(ngrok.connect(8501))


### 📈 Example Results
**Normal running:** risks ~0.2–0.4, no alerts.

**Spikes:** surges in vibration/temperature push risk ≥ threshold.

**Drift injection:** PSI rises, flagged in drift tab.

**Metrics (sample run):**
| Metric          | Value                    |
| --------------- | ------------------------ |
| Avg Risk        | 0.42                     |
| Max Risk        | 0.98 (S03)               |
| Alerts Raised   | 7 in last 10 mins        |
| PSI (Vibration) | 0.27 → significant drift |

## 📂 Project Structure
├── rt_maint.py              # Main Streamlit app
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── screenshots/             # Saved screenshots for README
