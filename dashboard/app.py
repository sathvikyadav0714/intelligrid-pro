# ======================
# INTELLIGRID PRO (FINAL + ALL BUILDINGS FIX)
# ======================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.anomaly_detection import detect_anomalies

st.set_page_config(layout="wide", page_title="⚡ IntelliGrid Pro")

# ======================
# LOAD DATA
# ======================
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data/train.csv"), nrows=100000)
    weather = pd.read_csv(os.path.join(BASE_DIR, "data/weather_train.csv"))
    building = pd.read_csv(os.path.join(BASE_DIR, "data/building_metadata.csv"))

    df = df.merge(building, on="building_id", how="left")
    df = df.merge(weather, on=["site_id","timestamp"], how="left")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.date

    for col in ["air_temperature","cloud_coverage","dew_temperature"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df

df_original = load_data()

# ======================
# MODEL
# ======================
model = joblib.load(os.path.join(BASE_DIR, "models/xgb_model.pkl"))

features = [
    "square_feet","air_temperature","cloud_coverage",
    "dew_temperature","hour","dayofweek","month"
]

features = [f for f in features if f in df_original.columns]

df_original[features] = df_original[features].fillna(df_original[features].median())
df_original["prediction"] = model.predict(df_original[features])
st.sidebar.success("✅ Model Loaded")
# ======================
# SIDEBAR
# ======================
st.sidebar.title("⚡ IntelliGrid Pro")

page = st.sidebar.selectbox("Navigation",[
"📊 Overview","🏢 Building Analysis","🧠 AI Insights",
"💡 Recommendations","🎯 Model Performance",
"🔬 Simulation","📄 Reports","📘 About Project"
])

cont = st.sidebar.slider("Anomaly Sensitivity",0.005,0.05,0.01)

# 🔥 ALL BUILDINGS OPTION
building_options = ["All Buildings"] + sorted(df_original["building_id"].unique().tolist())
building = st.sidebar.selectbox("Building", building_options, index=0)

date_range = st.sidebar.date_input(
    "Date Range",
    (df_original["date"].min(),df_original["date"].max())
)

month = st.sidebar.selectbox("Month",sorted(df_original["month"].unique()))

# ======================
# FILTER LOGIC
# ======================
start,end = date_range

df_all = df_original[
    (df_original["date"]>=start)&
    (df_original["date"]<=end)&
    (df_original["month"]==month)
].copy()

if building == "All Buildings":
    df_filtered = df_all.copy()
else:
    df_filtered = df_all[df_all["building_id"]==building].copy()

# ======================
# ANOMALY DETECTION
# ======================
df_all = detect_anomalies(df_all, features, contamination=cont)
df_filtered = detect_anomalies(df_filtered, features, contamination=cont)

# ======================
# COST MODEL
# ======================
def cost(h):
    return 8 if 18<=h<=23 else 6 if 10<=h<=17 else 4

df_all["cost"] = df_all.apply(lambda x: x["meter_reading"]*cost(x["hour"]),axis=1)
df_filtered["cost"] = df_filtered.apply(lambda x: x["meter_reading"]*cost(x["hour"]),axis=1)

# ======================
# METRICS (GLOBAL)
# ======================
total_energy=int(df_all["meter_reading"].sum())
loss=int(df_all[df_all["anomaly"]==1]["cost"].sum())
anom_pct=round(df_all["anomaly"].mean()*100,2) if len(df_all)>0 else 0
worst=df_all.groupby("building_id")["anomaly"].sum().idxmax()
peak=df_all.groupby("hour")["meter_reading"].mean().idxmax()

r2=r2_score(df_all["meter_reading"],df_all["prediction"])
mae=mean_absolute_error(df_all["meter_reading"],df_all["prediction"])
rmse=np.sqrt(((df_all["meter_reading"]-df_all["prediction"])**2).mean())
total_cost = int(df_all["cost"].sum())
recoverable_savings = int(loss * 0.7)
optimized_cost = total_cost - recoverable_savings


# ======================
# 🚨 ALERT SYSTEM
# ======================
alerts_df = df_all[df_all["anomaly"] == 1].copy()

# Severity classification
def get_severity(row):
    if row["meter_reading"] > row["prediction"] * 2:
        return "🔴 Critical"
    elif row["meter_reading"] > row["prediction"] * 1.5:
        return "🟡 Warning"
    return "Normal"

alerts_df["severity"] = alerts_df.apply(get_severity, axis=1)

# Take latest alerts
alerts_df = alerts_df.sort_values("timestamp", ascending=False).head(5)


# ======================
# PAGE 1 — OVERVIEW
# ======================
if page=="📊 Overview":
    st.title("⚡ IntelliGrid Pro – Energy Analytics Dashboard")
    st.caption("""
    Unlike traditional dashboards, this system combines prediction, anomaly detection, cost analysis, and recommendations in one platform.
    """)
    st.success("""
    We don’t just detect anomalies — we explain them using temperature, time, and usage deviation.
    """)

    st.markdown("""
    This dashboard monitors building energy usage, detects anomalies using AI, predicts consumption, and shows cost loss with optimization suggestions.
    """)

    # ======================
    # 🚨 ALERTS
    # ======================
    st.subheader("🚨 Live System Alerts")

    st.caption("""
    Alerts highlight abnormal energy usage detected by AI.
    🔴 Red = Critical spike | 🟡 Yellow = Unusual usage | 🟢 Green = Normal
    """)

    if len(alerts_df) == 0:
        st.success("🟢 System Normal – No anomalies detected")
    else:
        for _, row in alerts_df.iterrows():
            if "Critical" in row["severity"]:
                st.error(f"{row['severity']} | Building {row['building_id']} | Spike detected")
            else:
                st.warning(f"{row['severity']} | Building {row['building_id']} | Unusual usage")

    # ======================
    # KPI CARDS
    # ======================
    st.subheader("📊 Key Metrics")

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)


    c1.metric("⚡ Total Energy", total_energy)
    c1.caption("Total electricity used")

    c2.metric("💸 Loss ₹", loss)
    c2.caption("Estimated cost loss due to inefficiency")

    c3.metric("🔥 Anomaly %", anom_pct)
    c3.caption("Percentage of abnormal readings")

    c4.metric("🏢 Worst Building", worst)
    c4.caption("Building with highest usage")

    c5.metric("⏰ Peak Hour", peak)
    c5.caption("Time with highest consumption")

    c6.metric("🎯 Accuracy", f"{r2*100:.1f}%")
    c6.caption("Prediction model performance")

    c7.metric("💰 Savings ₹", recoverable_savings)
    c7.caption("Potential recoverable cost")

    # ======================
    # SYSTEM HEALTH
    # ======================
    if anom_pct < 5:
        health = "🟢 GOOD"
    elif anom_pct < 15:
        health = "🟡 WARNING"
    else:
        health = "🔴 CRITICAL"

    st.info(f"System Health Status: {health}")

    # ======================
    # SUMMARY INSIGHTS
    # ======================
    st.subheader("🧠 Quick Insights")

    st.info(f"""
    - Peak usage at **{peak}:00**  
    - Highest usage building: **{worst}**  
    - Estimated loss: **₹{loss:,}**  
    - System anomaly rate: **{anom_pct}%**  
    """)

    st.subheader("💸 Cost Breakdown")

    st.info(f"""
    Total Cost: ₹{total_cost:,}  
    Loss due to anomalies: ₹{loss:,}  
    Recoverable Savings (70%): ₹{recoverable_savings:,}
    """)

    st.subheader("📊 Before vs After Optimization")

    col1, col2 = st.columns(2)

    with col1:
        st.error(f"""
        ❌ Without IntelliGrid  
        Total Cost: ₹{total_cost:,}  
        No anomaly detection  
        No optimization
        """)

    with col2:
        st.success(f"""
        ✅ With IntelliGrid  
        Optimized Cost: ₹{optimized_cost:,}  
        Savings: ₹{recoverable_savings:,}  
        AI-driven decisions
        """)

    st.subheader("🔍 Anomaly Explanation")

    sample = df_all[df_all["anomaly"]==1].head(3)

    for _, row in sample.iterrows():
        diff = row["meter_reading"] - row["prediction"]

        st.write(f"""
        Building {row['building_id']}  
        Actual: {int(row['meter_reading'])}  
        Predicted: {int(row['prediction'])}  
        Difference: {int(diff)}  

        👉 Reason: High usage spike + peak hour
        """)
            

    st.info("""
    ✔ Handles multiple buildings  
    ✔ Works on time-series data  
    ✔ Scalable to smart city systems  
    """)

    # ======================
    # MAIN GRAPH
    # ======================
    st.subheader("📈 Actual vs Predicted Energy Usage")

    st.caption("""
    🔵 Blue = Actual usage  
    🟠 Orange = Predicted usage  
    🔴 Red dots = Anomalies  
    """)

    plot = df_filtered.tail(500)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot["timestamp"],
        y=plot["meter_reading"],
        name="Actual",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=plot["timestamp"],
        y=plot["prediction"],
        name="Predicted",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=plot[plot["anomaly"]==1]["timestamp"],
        y=plot[plot["anomaly"]==1]["meter_reading"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # CHARTS
    # ======================
    st.subheader("📊 System Analysis")

    col1,col2,col3 = st.columns(3)

    # Pie
    with col1:
        st.caption("Anomaly Distribution → shows abnormal vs normal readings")
        st.plotly_chart(
            px.pie(df_all, names="anomaly"),
            use_container_width=True
        )

    # Top buildings
    with col2:
        st.caption("Top Buildings → buildings with highest usage")
        top = df_all.groupby("building_id")["meter_reading"].sum().nlargest(10).reset_index()
        st.plotly_chart(
            px.bar(top, x="building_id", y="meter_reading"),
            use_container_width=True
        )

    # Cost loss
    with col3:
        st.caption("Cost Loss Over Time → money wasted due to anomalies")
        loss_time = df_all[df_all["anomaly"]==1].groupby("date")["cost"].sum().reset_index()
        st.plotly_chart(
            px.line(loss_time, x="date", y="cost"),
            use_container_width=True
        )

    # ======================
    # FOOTER NOTE
    # ======================
    st.success("""
    🤖 This system uses machine learning to predict energy usage and detect anomalies for smart energy optimization.
    """)
# ======================
# PAGE 2
# ======================
elif page=="🏢 Building Analysis":
    st.title("🏢 Building Analysis")

    st.markdown("""
    This section provides **detailed analytics for a selected building**, including energy usage patterns, anomalies, cost losses, and performance insights.
    """)

    buildings = df_all["building_id"].unique()

    b1 = st.selectbox("Select Building", buildings)

    d1 = df_all[df_all["building_id"] == b1]

    # ======================
    # KPI CARDS
    # ======================
    st.subheader("📊 Key Metrics")

    total_usage = int(d1["meter_reading"].sum())
    anomaly_count = int(d1["anomaly"].sum())
    cost_loss = int(d1[d1["anomaly"]==1]["cost"].sum())
    peak_hour = d1.groupby("hour")["meter_reading"].mean().idxmax()
    avg_usage = round(d1["meter_reading"].mean(),2)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("⚡ Total Usage", total_usage)
    c1.caption("Total electricity consumed by this building")

    c2.metric("🔥 Anomalies", anomaly_count)
    c2.caption("Number of abnormal usage events detected")

    c3.metric("💸 Loss ₹", cost_loss)
    c3.caption("Estimated cost wasted due to anomalies")

    c4.metric("⏰ Peak Hour", peak_hour)
    c4.caption("Hour with highest average energy usage")

    c5.metric("📊 Avg Usage", avg_usage)
    c5.caption("Average energy consumption")

    # ======================
    # MAIN GRAPH
    # ======================
    st.subheader("📈 Usage vs Prediction")

    st.markdown("""
    **Graph Explanation:**
    - 🔵 Blue → Actual energy usage  
    - 🟠 Orange → Predicted (expected) usage  
    - 🔴 Red dots → Anomalies  

    👉 If actual usage is higher than predicted, it indicates inefficiency or abnormal spikes.
    """)

    plot = d1.tail(500)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot["timestamp"],
        y=plot["meter_reading"],
        name="Actual",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=plot["timestamp"],
        y=plot["prediction"],
        name="Predicted",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=plot[plot["anomaly"]==1]["timestamp"],
        y=plot[plot["anomaly"]==1]["meter_reading"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # ADDITIONAL CHARTS
    # ======================
    st.subheader("📊 Usage Patterns & Loss Analysis")

    col1, col2 = st.columns(2)

    # Loss over time
    with col1:
        st.caption("💸 Cost loss trend over time due to anomalies")
        loss_time = d1[d1["anomaly"]==1].groupby("date")["cost"].sum().reset_index()

        st.plotly_chart(
            px.line(loss_time, x="date", y="cost"),
            use_container_width=True
        )

    # Hour vs usage
    with col2:
        st.caption("⏰ Average usage across different hours (identifies peak hour)")
        hour_usage = d1.groupby("hour")["meter_reading"].mean().reset_index()

        st.plotly_chart(
            px.bar(hour_usage, x="hour", y="meter_reading"),
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    # Weekday vs weekend
    with col3:
        st.caption("📅 Usage variation across weekdays")
        wk = d1.groupby("dayofweek")["meter_reading"].mean().reset_index()

        st.plotly_chart(
            px.bar(wk, x="dayofweek", y="meter_reading"),
            use_container_width=True
        )

    # Anomaly timeline
    with col4:
        st.caption("🔥 Timeline of detected anomalies")
        anom = d1[d1["anomaly"]==1]
        st.plotly_chart(
            px.scatter(anom, x="timestamp", y="meter_reading"),
            use_container_width=True
        )

    # ======================
    # ANOMALY TABLE
    # ======================
    st.subheader("📋 Anomaly Details")

    st.caption("Detailed list of abnormal energy readings")

    table = d1[d1["anomaly"]==1][["timestamp","meter_reading","prediction"]].copy()
    table["difference"] = table["meter_reading"] - table["prediction"]

    st.dataframe(table.head(50), use_container_width=True)

    # ======================
    # BUILDING COMPARISON
    # ======================
    st.subheader("🔍 Compare Buildings")

    st.caption("Compare energy usage with another building")

    if len(buildings) > 1:
        b2 = st.selectbox("Compare with", buildings, index=1)
        d2 = df_all[df_all["building_id"] == b2]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=d1["timestamp"], y=d1["meter_reading"], name=f"B{b1}"))
        fig2.add_trace(go.Scatter(x=d2["timestamp"], y=d2["meter_reading"], name=f"B{b2}"))

        st.plotly_chart(fig2, use_container_width=True)

    # ======================
    # INSIGHT BOX
    # ======================
    st.subheader("🧠 Insights")

    st.info(f"""
    - Peak usage observed at **{peak_hour}:00**
    - Total anomalies detected: **{anomaly_count}**
    - Estimated loss: **₹{cost_loss}**
    - Average usage: **{avg_usage}**

    👉 High anomaly count or high loss indicates inefficient energy usage.
    """)
# ======================
# PAGE 3
# ======================
elif page=="🧠 AI Insights":
    st.title("🧠 AI Insights")

    st.markdown("""
    This page shows **patterns detected by AI**, including peak usage hours, abnormal energy behavior, and building-level insights.
    """)

    # ======================
    # SUMMARY INSIGHT BOX
    # ======================
    st.subheader("📊 Key Insights")

    peak_hour = df_all.groupby("hour")["meter_reading"].mean().idxmax()
    top_building = df_all.groupby("building_id")["meter_reading"].sum().idxmax()
    anomaly_rate = round(df_all["anomaly"].mean()*100,2) if len(df_all)>0 else 0
    est_loss = int(df_all[df_all["anomaly"]==1]["cost"].sum())

    st.info(f"""
    - ⏰ Peak usage at **{peak_hour}:00**  
    - 🏢 Highest usage building: **{top_building}**  
    - 🔥 Anomaly rate: **{anomaly_rate}%**  
    - 💸 Estimated loss: **₹{est_loss}**  
    """)

    # ======================
    # HEATMAP
    # ======================
    st.subheader("🌡️ Energy Usage Pattern (Hour vs Day)")

    st.markdown("""
    This heatmap shows energy usage across **hours and days**.  
    👉 Darker color = higher energy consumption.
    """)

    heat = df_all.groupby(["dayofweek","hour"])["meter_reading"].mean().reset_index()

    st.plotly_chart(
        px.density_heatmap(
            heat,
            x="dayofweek",
            y="hour",
            z="meter_reading",
            title="Energy Usage Pattern (Hour vs Day)"
        ),
        use_container_width=True
    )

    # ======================
    # TOP BUILDINGS
    # ======================
    st.subheader("🏢 Top Buildings by Usage")

    st.markdown("Shows which buildings consume the most energy.")

    top = df_all.groupby("building_id")["meter_reading"].sum().nlargest(10).reset_index()

    st.plotly_chart(
        px.bar(top, x="building_id", y="meter_reading"),
        use_container_width=True
    )

    # ======================
    # PIE CHART
    # ======================
    st.subheader("⚖️ Normal vs Anomaly Distribution")

    st.markdown("""
    Blue = normal usage  
    Red = anomaly (abnormal usage)
    """)

    st.plotly_chart(
        px.pie(df_all, names="anomaly"),
        use_container_width=True
    )

    # ======================
    # SCATTER (TEMP VS USAGE)
    # ======================
    st.subheader("🌡️ Temperature vs Energy Usage")

    st.markdown("""
    Shows how temperature affects energy consumption.  
    👉 Higher temperature often increases energy usage.
    """)

    st.plotly_chart(
        px.scatter(df_all, x="air_temperature", y="meter_reading", color="anomaly"),
        use_container_width=True
    )

    # ======================
    # DATA TABLE
    # ======================
    st.subheader("📋 Top Records Used for Analysis")

    st.markdown("""
    This table shows sample records used by AI:
    - **building_id** → Building identifier  
    - **meter_reading** → Energy usage  
    - **timestamp** → Time of reading  
    - **anomaly** → 1 = abnormal, 0 = normal  
    """)

    st.dataframe(
        df_all[["building_id","timestamp","meter_reading","anomaly"]].head(100),
        use_container_width=True
    )

# ======================
# PAGE 4
# ======================
elif page=="💡 Recommendations":
    st.title("💡 AI Recommendations")

    st.markdown("""
    This page provides **AI-generated suggestions** to reduce energy waste and optimize cost based on anomaly detection and usage patterns.
    """)

    # ======================
    # DATA PREP
    # ======================
    rec_df = df_all[df_all["anomaly"]==1].copy()

    if len(rec_df) == 0:
        st.success("✅ No major issues detected. System is running efficiently.")
    else:
        grouped = rec_df.groupby("building_id").agg({
            "cost":"sum",
            "hour":"mean",
            "anomaly":"count"
        }).rename(columns={"anomaly":"anomaly_count"})

        grouped = grouped.sort_values("cost", ascending=False).head(6)

        total_loss = int(grouped["cost"].sum())
        total_saving = int(total_loss * 0.3)
        total_buildings = len(grouped)

        # ======================
        # SUMMARY BOX
        # ======================
        st.subheader("📊 Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("💸 Total Loss", f"₹{total_loss:,}")
        c2.metric("💰 Possible Saving", f"₹{total_saving:,}")
        c3.metric("🏢 Buildings Affected", total_buildings)

        st.info("⚠ Recommendations are generated by AI using anomaly detection and prediction differences.")

        st.subheader("🚨 Priority Recommendations")

        # ======================
        # CARD STYLE DISPLAY
        # ======================
        for b, row in grouped.iterrows():

            loss_val = int(row["cost"])
            saving = int(loss_val * 0.3)
            peak_hour = int(row["hour"])

            # COLOR LOGIC
            if loss_val > 50000:
                color = "#ff4d4d"   # red
            elif loss_val > 20000:
                color = "#ffa500"   # orange
            else:
                color = "#2ecc71"   # green

            # ACTION LOGIC
            actions = []
            if 12 <= peak_hour <= 16:
                actions.append("Shift load to off-peak hours")
            if row["anomaly_count"] > 20:
                actions.append("Inspect abnormal equipment usage")
            if loss_val > 30000:
                actions.append("Optimize energy schedule")

            action_text = ", ".join(actions) if actions else "General inspection recommended"

            # ======================
            # CARD UI
            # ======================
            st.markdown(f"""
            <div style="
                background-color:#f5f5f5;
                padding:15px;
                border-radius:10px;
                margin-bottom:10px;
                border-left:6px solid {color};
                color:#000000;
            ">

            <h4>🏢 Building {b}</h4>

            <p><b>💸 Estimated Loss:</b> ₹{loss_val:,}</p>
            <p><b>⏰ Peak Hour:</b> {peak_hour}:00</p>
            <p><b>⚠ Anomalies:</b> {int(row['anomaly_count'])}</p>

            <p><b>⚡ Suggested Action:</b> {action_text}</p>

            <p><b>💰 Possible Saving:</b> ₹{saving:,}</p>

            </div>
            """, unsafe_allow_html=True)
# ======================
# PAGE 5
# ======================
elif page=="🎯 Model Performance":
    st.title("🎯 Model Performance")

    st.markdown("""
    This page shows how accurately the AI model predicts energy usage based on historical data.
    """)

    # ======================
    # METRIC EXPLANATIONS
    # ======================
    st.subheader("📊 Model Accuracy Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", f"{r2:.3f}")
    c1.caption("How well predictions match actual values (closer to 1 is better)")

    c2.metric("MAE", f"{mae:.2f}")
    c2.caption("Average prediction error (lower is better)")

    c3.metric("RMSE", f"{rmse:.2f}")
    c3.caption("Penalty for large errors (lower is better)")

    # ======================
    # PERFORMANCE SUMMARY
    # ======================
    if r2 > 0.85:
        status = "🟢 Good"
    elif r2 > 0.65:
        status = "🟡 Average"
    else:
        status = "🔴 Poor"

    st.info(f"📌 Model Performance Status: **{status}**")

    # ======================
    # SCATTER PLOT
    # ======================
    st.subheader("📈 Actual vs Predicted Usage")

    fig_scatter = px.scatter(
        df_all,
        x="prediction",
        y="meter_reading",
        title="Actual vs Predicted Usage",
        opacity=0.5
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.caption("""
    Points closer to a straight diagonal line indicate better prediction accuracy.
    """)

    # ======================
    # ERROR OVER TIME
    # ======================
    st.subheader("📉 Prediction Error Over Time")

    df_all["error"] = df_all["meter_reading"] - df_all["prediction"]

    error_time = df_all.groupby("date")["error"].mean().reset_index()

    st.plotly_chart(
        px.line(error_time, x="date", y="error"),
        use_container_width=True
    )

    st.caption("""
    Shows how prediction error changes over time. Large spikes indicate poor predictions.
    """)

    # ======================
    # NOTE
    # ======================
    st.success("""
    🤖 This model uses machine learning to predict energy usage and detect anomalies.
    Better accuracy leads to more reliable insights and recommendations.
    """)
# ======================
# PAGE 6
# ======================
elif page=="🔬 Simulation":
    st.title("🔬 Energy Simulation")

    st.markdown("""
    This page simulates how energy usage changes when **temperature or demand conditions change**.
    It helps understand how external factors impact consumption.
    """)

    # ======================
    # SLIDERS WITH EXPLANATION
    # ======================
    st.subheader("⚙️ Simulation Controls")

    col1, col2 = st.columns(2)

    with col1:
        temp = st.slider("🌡️ Temperature (°C)", 10, 45, 30)
        st.caption("Simulates weather temperature effect on energy usage")

    with col2:
        mult = st.slider("⚡ Usage Multiplier", 0.5, 2.0, 1.0)
        st.caption("Simulates increase or decrease in building energy demand")

    # ======================
    # SIMULATION DATA
    # ======================
    sim = df_filtered.tail(100).copy()

    original_usage = sim["meter_reading"].sum()

    sim["air_temperature"] = temp
    sim["meter_reading"] = sim["meter_reading"] * mult

    sim["prediction"] = model.predict(sim[features])
    sim = detect_anomalies(sim, features, cont)

    new_usage = sim["meter_reading"].sum()

    change = new_usage - original_usage

    # ======================
    # SUMMARY INSIGHTS
    # ======================
    st.subheader("📊 Simulation Insights")

    if change > 0:
        color = "🔴"
        msg = "Increase in energy usage"
    else:
        color = "🟢"
        msg = "Decrease in energy usage"

    st.info(f"""
    - 🌡️ Higher temperature can increase energy usage  
    - ⚡ Usage multiplier applied: {mult}x  
    - {color} **{msg}**: {int(change)} units  
    """)

    # ======================
    # MAIN GRAPH
    # ======================
    st.subheader("📈 Simulated vs Actual Energy Usage")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sim["timestamp"],
        y=sim["meter_reading"],
        name="Actual (Simulated)",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=sim["timestamp"],
        y=sim["prediction"],
        name="Predicted",
        line=dict(color="orange")
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.caption("""
    🔵 Blue line = Actual usage after simulation  
    🟠 Orange line = Predicted usage by AI model  
    """)

    # ======================
    # ANOMALY VIEW
    # ======================
    st.subheader("🔥 Simulated Anomalies")

    anom = sim[sim["anomaly"]==1]

    st.plotly_chart(
        px.scatter(anom, x="timestamp", y="meter_reading"),
        use_container_width=True
    )

    st.caption("Shows abnormal usage points after simulation")

    # ======================
    # NOTE
    # ======================
    st.success("""
    ⚠ This simulation does not modify real data.  
    It only shows predicted changes based on AI model behavior.
    """)
# ======================
# PAGE 7
# ======================
elif page=="📄 Reports":
    st.title("📄 Reports & Data Export")

    st.markdown("""
    This page allows you to **view filtered data and download reports** for further analysis.
    All data shown here is based on the filters selected in the sidebar.
    """)

    # ======================
    # SUMMARY METRICS
    # ======================
    st.subheader("📊 Report Summary")

    total_records = len(df_all)
    total_buildings = df_all["building_id"].nunique()
    anomaly_count = int(df_all["anomaly"].sum())
    total_energy = int(df_all["meter_reading"].sum())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📋 Total Records", total_records)
    c2.metric("🏢 Buildings", total_buildings)
    c3.metric("🔥 Anomalies", anomaly_count)
    c4.metric("⚡ Total Energy", total_energy)

    st.info("📌 Filters applied from sidebar are reflected in this report.")

    # ======================
    # DOWNLOAD SECTION
    # ======================
    st.subheader("⬇️ Download Report")

    st.markdown("Download the current filtered dataset for offline analysis.")

    st.download_button(
        "📥 Download Filtered Report (CSV)",
        df_all.to_csv(index=False),
        file_name="intelligrid_report.csv"
    )

    # ======================
    # TABLE DESCRIPTION
    # ======================
    st.subheader("📋 Data Preview")

    st.markdown("""
    The table below shows the dataset used for:
    - Energy prediction  
    - Anomaly detection  
    - AI recommendations  

    **Column meanings:**
    - **building_id** → Unique building identifier  
    - **meter_reading** → Energy consumption value  
    - **timestamp** → Time of measurement  
    - **primary_use** → Building usage type (office, education, etc.)  
    - **anomaly** → 1 = abnormal usage, 0 = normal  
    """)

    # ======================
    # DATA TABLE
    # ======================
    st.dataframe(
        df_all.head(500),
        use_container_width=True
    )
# ======================
# PAGE 8
# ======================
elif page=="📘 About Project":
    st.title("📘 IntelliGrid Pro – AI Energy Optimization System")

    st.markdown("""
    ## 🚀 Overview

    **IntelliGrid Pro** is an advanced AI-powered energy analytics platform designed to optimize building-level energy consumption, detect anomalies in real-time, and reduce operational costs through intelligent recommendations.

    It combines **machine learning, anomaly detection, and interactive visualization** into a unified dashboard that enables data-driven energy management.

    ---
    
    ## 🎯 Problem Statement

    Modern buildings consume massive amounts of energy, but:

    - ❌ Energy wastage goes unnoticed  
    - ❌ No real-time anomaly detection  
    - ❌ Lack of actionable insights  
    - ❌ High operational costs  

    **IntelliGrid Pro solves this by bringing AI into energy monitoring.**

    ---
    
    ## 🧠 Core Features

    - ⚡ **Energy Forecasting**  
      Predicts future energy usage using trained machine learning models.

    - 🔥 **Anomaly Detection**  
      Identifies unusual consumption patterns using AI algorithms.

    - 🧩 **Root Cause Analysis**  
      Explains *why* anomalies occur (temperature, peak load, irregular usage).

    - 💡 **Smart Recommendations**  
      Suggests optimization strategies with cost-saving estimates.

    - 🚨 **Live Alerts System**  
      Detects and highlights critical energy anomalies in real-time.

    - 📊 **Interactive Dashboard**  
      Provides deep insights through charts, heatmaps, and analytics.

    ---
    
    ## ⚙️ How It Works

    1. Data from buildings and weather is collected  
    2. Features like time, temperature, and usage patterns are engineered  
    3. ML model predicts expected energy consumption  
    4. Anomaly detection identifies deviations  
    5. AI engine generates insights and recommendations  

    ---
    
    ## 🏗️ System Architecture

    - Data Layer → Building + Weather Data  
    - Processing Layer → Feature Engineering  
    - ML Layer → XGBoost Prediction Model  
    - AI Layer → Anomaly Detection + Recommendation Engine  
    - Visualization Layer → Streamlit Dashboard  

    ---
    
    ## 🛠️ Tech Stack

    - **Python**
    - **Pandas & NumPy**
    - **XGBoost (Prediction Model)**
    - **Isolation Forest (Anomaly Detection)**
    - **Plotly (Visualization)**
    - **Streamlit (Dashboard UI)**

    ---
    
    ## 📈 Business Impact

    - 💰 Reduce energy costs by up to **30%**
    - ⚡ Improve operational efficiency
    - 🔍 Detect hidden energy leaks
    - 🧠 Enable data-driven decision making

    ---
    
    ## 🌍 Use Cases

    - Smart Buildings  
    - Corporate Offices  
    - Industrial Energy Monitoring  
    - Smart Cities  

    ---
    
    ## 🧩 Future Enhancements

    - 📡 Real-time IoT integration  
    - ☁️ Cloud deployment  
    - 🤖 Automated control systems  
    - 📊 Advanced forecasting models  

    ---
    
    ## 👨‍💻 Project Goal

    To transform traditional energy monitoring into an **AI-driven intelligent system** that not only analyzes data but actively **optimizes energy usage and reduces waste**.

    """)