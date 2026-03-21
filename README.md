# ⚡ IntelliGrid Pro – AI Energy Forecasting & Anomaly Detection

> Predict. Detect. Optimize.  
> An AI-powered energy analytics platform for forecasting consumption, detecting anomalies, and optimizing building performance.

---

## 🚀 Project Overview

**IntelliGrid Pro** is a full-stack machine learning based energy monitoring system designed to analyze building energy usage, predict future consumption, detect abnormal patterns, and generate intelligent recommendations to reduce energy waste.

The platform converts raw energy data into **actionable insights, alerts, and cost-saving decisions** using advanced ML models and interactive dashboards.

---

## 🎯 Problem Statement

Energy inefficiency in buildings leads to high operational cost and hidden waste.

Common issues:
- No real-time anomaly detection
- No forecasting of energy usage
- Lack of intelligent recommendations
- Difficult to analyze large energy datasets
- No simulation before applying changes

**IntelliGrid Pro solves this using AI-driven analytics.**

---

## 🧠 Key Features

### 🔮 Energy Forecasting
- Predicts future energy consumption using **XGBoost**
- Learns patterns from time, weather, and building metadata
- Helps plan energy usage efficiently

### 🔥 Anomaly Detection
- Uses **Isolation Forest**
- Detects abnormal spikes in energy usage
- Helps identify faults, leaks, or inefficiencies

### 🚨 Smart Alert System
- Classifies anomalies as warning / critical
- Real-time alert indicators
- Helps prioritize issues

### 💡 AI Recommendation Engine
- Suggests actions to reduce energy waste
- Example:
  - HVAC optimization
  - Load shifting
  - Temperature adjustment
- Shows estimated cost savings

### 🔬 Simulation Engine
- Simulates energy usage under different conditions
- Temperature change
- Demand change
- Usage variation

Helps test before real-world implementation.

### 📊 Interactive Dashboard
Built with:
- Streamlit
- Plotly

Includes:
- KPIs
- Charts
- Heatmaps
- Alerts
- Reports

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|------------|
| Overview | KPIs, total energy, alerts, trends |
| Building Analysis | Individual building performance |
| AI Insights | Heatmaps, anomaly patterns |
| Recommendations | AI cost saving suggestions |
| Model Performance | ML metrics |
| Simulation | Scenario testing |
| Reports | Exportable summaries |
| About | Project details |

---

## ⚙️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost |
| Anomaly Detection | Isolation Forest |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Dataset | ASHRAE Energy Prediction |

---

## 🧪 Workflow

1. Load building, weather, and energy data  
2. Perform feature engineering  
3. Train forecasting model (XGBoost)  
4. Predict expected energy usage  
5. Detect anomalies using Isolation Forest  
6. Generate insights & recommendations  
7. Visualize results in dashboard  

---

## 📈 Business Impact

- Reduce energy cost up to 30%
- Detect hidden inefficiencies
- Improve building performance
- Enable data-driven decisions
- Useful for smart cities & IoT systems

---

## 📁 Dataset

Dataset not included due to size.

Download from Kaggle:

https://www.kaggle.com/competitions/ashrae-energy-prediction

Place files inside:

```
data/
```

Required files:

```
train.csv
weather_train.csv
building_metadata.csv
```

---

## ▶️ How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Run project

```bash
streamlit run app.py
```

---

## 📷 Screenshots

```
screenshots/overview.png
screenshots/ai_insights.png
screenshots/building_analysis.png
screenshots/ai_recommendations.png
screenshots/energy_simulation.png
screenshots/model_performance.png
screenshots/reports.png
screenshots/about.png
```

---

## 🚀 Future Improvements

- Real-time IoT data integration
- Cloud deployment (AWS / GCP)
- Deep learning forecasting
- Auto energy control system
- Multi-building monitoring

---

## 👨‍💻 Author

Sathvik  

---
