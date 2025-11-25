import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# PAGE CONFIG
st.set_page_config(
    page_title="HealthFlow",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TITLE AND HEADER
st.title("🏥 HealthFlow - Hospital Resource Predictor")
st.markdown("**AI-powered forecasting for Malaysian hospital capacity management**")
st.markdown("---")

# SIDEBAR - HOSPITAL CONTROLS
st.sidebar.header("🏥 Hospital Configuration")

hospital_name = st.sidebar.selectbox(
    "Select Hospital",
    ["Hospital Kuala Lumpur", "Hospital Selayang", "Hospital Umum Sarawak", "Hospital Queen Elizabeth"]
)

total_beds = st.sidebar.slider("Total Bed Capacity", 50, 500, 200)
prediction_days = st.sidebar.slider("Prediction Days", 3, 14, 7)

demo_mode = st.sidebar.radio(
    "Demo Scenario",
    ["Normal Operations", "Weekend Surge", "Seasonal Outbreak", "Emergency Crisis"]
)

# KEY METRICS
st.subheader("📊 Real-time Capacity Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Bed Occupancy", "84%", "+2%", delta_color="inverse")

with col2:
    st.metric("ICU Availability", "4/20 beds", "-2 beds", delta_color="inverse")

with col3:
    st.metric("Predicted Peak", "96%", "in 3 days")

with col4:
    st.metric("Staff Readiness", "78%", "-5% shortage", delta_color="inverse")

# MAIN PREDICTION CHART
st.subheader("📈 Bed Occupancy Forecast")

def get_prediction_data(scenario, days=7):
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    
    if scenario == "Normal Operations":
        occupancy = [75, 78, 76, 74, 72, 70, 68]
    elif scenario == "Weekend Surge":
        occupancy = [75, 78, 82, 85, 88, 92, 85]
    elif scenario == "Seasonal Outbreak":
        occupancy = [80, 85, 90, 94, 96, 95, 92]
    else:  # Emergency Crisis
        occupancy = [85, 90, 95, 98, 99, 97, 94]
    
    # Extend or truncate based on prediction_days
    occupancy = occupancy[:days]
    if len(occupancy) < days:
        occupancy.extend([occupancy[-1]] * (days - len(occupancy)))
    
    return pd.DataFrame({'date': dates, 'occupancy_rate': occupancy})

df = get_prediction_data(demo_mode, prediction_days)

# Create chart with alert zones
fig = px.line(
    df, 
    x='date', 
    y='occupancy_rate',
    title=f'Bed Occupancy Forecast - {demo_mode}',
    labels={'occupancy_rate': 'Occupancy Rate (%)', 'date': 'Date'}
)

# Add critical thresholds
fig.add_hrect(y0=90, y1=100, fillcolor="red", opacity=0.2, 
              annotation_text="CRITICAL ZONE", annotation_position="top left")
fig.add_hrect(y0=80, y1=90, fillcolor="orange", opacity=0.2,
              annotation_text="WARNING ZONE")

fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# ALERT SYSTEM
st.subheader("🚨 Alert & Action Center")

max_occupancy = df['occupancy_rate'].max()

if max_occupancy >= 95:
    st.error("""
    🚨 **CRITICAL ALERT**: Capacity Crisis Imminent
    - Predicted occupancy: 95%+
    - Time to crisis: 2-3 days
    - **IMMEDIATE ACTION REQUIRED**
    """)
    
    with st.expander("🚑 Emergency Protocol Actions", expanded=True):
        st.write("""
        1. ✅ Activate all emergency beds
        2. ✅ Recall off-duty medical staff  
        3. ✅ Postpone non-urgent surgeries
        4. ✅ Coordinate with nearby hospitals
        5. ✅ Set up temporary treatment areas
        """)
        
elif max_occupancy >= 85:
    st.warning("""
    ⚠️ **HIGH ALERT**: Resource Strain Expected  
    - Predicted occupancy: 85-94%
    - **PROACTIVE PLANNING REQUIRED**
    """)
    
    with st.expander("📋 Recommended Preparations", expanded=True):
        st.write("""
        1. 📋 Prepare emergency bed inventory
        2. 📋 Schedule additional staff on standby
        3. 📋 Review patient discharge schedules
        4. 📋 Check medical supply levels
        """)
        
else:
    st.success("""
    ✅ **NORMAL OPERATIONS**
    - Predicted occupancy: <85%
    - Standard protocols sufficient
    """)

# RESOURCE PLANNING SECTION
st.subheader("👥 Staffing & Resource Forecast")

col1, col2 = st.columns(2)

with col1:
    st.write("**Nursing Staff Requirements**")
    staffing_data = pd.DataFrame({
        'Day': ['Today', 'Tomorrow', 'Day 3', 'Day 4', 'Day 5'][:prediction_days],
        'Required Nurses': [45, 48, 52, 58, 55][:prediction_days],
        'Scheduled Nurses': [42, 45, 48, 50, 48][:prediction_days]
    })
    st.dataframe(staffing_data, use_container_width=True)

with col2:
    st.write("**Equipment Availability**")
    equipment_data = pd.DataFrame({
        'Equipment': ['Ventilators', 'ICU Monitors', 'Emergency Beds', 'Oxygen Tanks'],
        'Available': [18, 22, 15, 40],
        'Required': [15, 25, 20, 35],
        'Status': ['✅ Sufficient', '⚠️ Low', '❌ Critical', '✅ Sufficient']
    })
    st.dataframe(equipment_data, use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown("*HealthFlow - Empowering Malaysian Healthcare with AI Predictions*")
