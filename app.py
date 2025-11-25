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

# SIDEBAR - DATA UPLOAD
st.sidebar.header("📁 Upload Hospital Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload hospital data with columns: date, admissions, discharges, bed_occupancy, icu_patients"
)

use_real_data = st.sidebar.checkbox("Use uploaded data", value=False)

# DATA PROCESSING FUNCTIONS
def process_data(uploaded_file, scenario, use_real_data=False):
    if use_real_data and uploaded_file is not None:
        try:
            # Use REAL uploaded data
            real_df = pd.read_csv(uploaded_file)
            
            # Convert date column if exists
            if 'date' in real_df.columns:
                real_df['date'] = pd.to_datetime(real_df['date'])
            
            st.success(f"✅ Using real dataset: {len(real_df)} records loaded!")
            
            # Basic analysis of real data
            st.sidebar.subheader("📊 Dataset Summary")
            st.sidebar.write(f"Records: {len(real_df)}")
            st.sidebar.write(f"Columns: {list(real_df.columns)}")
            
            return real_df
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return get_mock_data(scenario)
    else:
        # Use MOCK data
        st.info("📊 Using demo data - toggle checkbox to use uploaded file")
        return get_mock_data(scenario)

def get_mock_data(scenario):
    dates = [datetime.now() + timedelta(days=i) for i in range(7)]
    
    if scenario == "Normal Operations":
        occupancy = [75, 78, 76, 74, 72, 70, 68]
        icu_patients = [12, 13, 11, 10, 9, 8, 7]
    elif scenario == "Weekend Surge":
        occupancy = [75, 78, 82, 85, 88, 92, 85]
        icu_patients = [12, 14, 16, 18, 19, 20, 17]
    elif scenario == "Seasonal Outbreak":
        occupancy = [80, 85, 90, 94, 96, 95, 92]
        icu_patients = [15, 17, 19, 21, 22, 21, 20]
    else:  # Emergency Crisis
        occupancy = [85, 90, 95, 98, 99, 97, 94]
        icu_patients = [18, 19, 20, 22, 23, 22, 21]
    
    return pd.DataFrame({
        'date': dates, 
        'occupancy_rate': occupancy,
        'icu_patients': icu_patients,
        'admissions': [np.random.randint(40, 60) for _ in range(7)],
        'discharges': [np.random.randint(35, 55) for _ in range(7)]
    })

# PROCESS DATA
df = process_data(uploaded_file, demo_mode, use_real_data)

# REAL DATA ANALYSIS
if use_real_data and uploaded_file is not None:
    st.subheader("📈 Real Data Analysis")
    
    # Show sample of real data
    with st.expander("View Uploaded Data"):
        st.dataframe(df.head(10))
    
    # Basic statistics
    if len(df.columns) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                st.metric("Numerical Columns", len(numeric_columns))
        with col3:
            if 'date' in df.columns:
                date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)

# KEY METRICS - DYNAMIC CALCULATIONS
st.subheader("📊 Real-time Capacity Overview")

col1, col2, col3, col4 = st.columns(4)

# Current Bed Occupancy
with col1:
    if use_real_data and uploaded_file is not None:
        if 'bed_occupancy' in df.columns and len(df) > 0:
            current_occupancy = df['bed_occupancy'].iloc[-1]
            # Calculate change from previous day
            if len(df) > 1:
                prev_occupancy = df['bed_occupancy'].iloc[-2]
                change = current_occupancy - prev_occupancy
                change_text = f"{change:+.1f}%"
            else:
                change_text = "No previous data"
            st.metric("Current Bed Occupancy", f"{current_occupancy:.1f}%", change_text, delta_color="inverse")
        else:
            st.metric("Current Bed Occupancy", "N/A", "No occupancy data")
    else:
        # Demo mode - use first value from mock data
        current_occupancy = df['occupancy_rate'].iloc[0] if 'occupancy_rate' in df.columns else 84
        if len(df) > 1:
            prev_occupancy = df['occupancy_rate'].iloc[1] if 'occupancy_rate' in df.columns else current_occupancy
            change = current_occupancy - prev_occupancy
            change_text = f"{change:+.1f}%"
        else:
            change_text = "No change data"
        st.metric("Current Bed Occupancy", f"{current_occupancy}%", change_text, delta_color="inverse")

# ICU Availability
with col2:
    if use_real_data and uploaded_file is not None and 'icu_patients' in df.columns and len(df) > 0:
        current_icu = df['icu_patients'].iloc[-1]
        total_icu_beds = 20
        available_icu = total_icu_beds - current_icu
        
        if len(df) > 1:
            prev_icu = df['icu_patients'].iloc[-2]
            change_icu = current_icu - prev_icu
            change_text = f"{change_icu:+.0f} patients"
        else:
            change_text = "No previous data"
            
        st.metric("ICU Availability", f"{available_icu}/{total_icu_beds} beds", change_text, delta_color="inverse")
    else:
        # Demo mode
        if 'icu_patients' in df.columns and len(df) > 0:
            current_icu = df['icu_patients'].iloc[0]
            total_icu_beds = 20
            available_icu = total_icu_beds - current_icu
            
            if len(df) > 1:
                prev_icu = df['icu_patients'].iloc[1]
                change_icu = current_icu - prev_icu
                change_text = f"{change_icu:+.0f} patients"
            else:
                change_text = "No change data"
                
            st.metric("ICU Availability", f"{available_icu}/{total_icu_beds} beds", change_text, delta_color="inverse")
        else:
            st.metric("ICU Availability", "4/20 beds", "-2 patients", delta_color="inverse")

# Predicted/Historical Peak
with col3:
    if use_real_data and uploaded_file is not None and 'bed_occupancy' in df.columns and len(df) > 0:
        predicted_peak = df['bed_occupancy'].max()
        peak_index = df['bed_occupancy'].idxmax()
        current_index = len(df) - 1
        days_to_peak = peak_index - current_index
        
        if days_to_peak > 0:
            peak_text = f"in {days_to_peak} days"
        elif days_to_peak == 0:
            peak_text = "today"
        else:
            peak_text = f"{abs(days_to_peak)} days ago"
            
        st.metric("Historical Peak", f"{predicted_peak:.1f}%", peak_text)
    else:
        # Demo mode - calculate from mock data
        if 'occupancy_rate' in df.columns and len(df) > 0:
            predicted_peak = df['occupancy_rate'].max()
            peak_day = df['occupancy_rate'].idxmax() + 1
            st.metric("Predicted Peak", f"{predicted_peak}%", f"on day {peak_day}")
        else:
            st.metric("Predicted Peak", "96%", "in 3 days")

# Staff Readiness
with col4:
    if use_real_data and uploaded_file is not None and 'bed_occupancy' in df.columns and len(df) > 0:
        current_occupancy = df['bed_occupancy'].iloc[-1]
        # Staffing model: 1 nurse per 4 patients at 100% capacity
        nurses_needed = (current_occupancy / 100) * (total_beds / 4)
        nurses_available = 35  # Base staff + adjustments
        staff_ratio = (nurses_available / nurses_needed) * 100 if nurses_needed > 0 else 100
        shortage = nurses_needed - nurses_available
        
        if shortage > 1:
            shortage_text = f"{shortage:.0f} nurse shortage"
        elif shortage > 0:
            shortage_text = f"{shortage:.1f} nurse shortage"
        else:
            shortage_text = "Adequate staff"
            
        st.metric("Staff Readiness", f"{min(staff_ratio, 100):.0f}%", shortage_text, delta_color="inverse")
    else:
        # Demo mode
        if 'occupancy_rate' in df.columns and len(df) > 0:
            current_occupancy = df['occupancy_rate'].iloc[0]
            nurses_needed = (current_occupancy / 100) * (total_beds / 4)
            nurses_available = 35
            staff_ratio = (nurses_available / nurses_needed) * 100 if nurses_needed > 0 else 100
            shortage = nurses_needed - nurses_available
            
            if shortage > 0:
                shortage_text = f"{shortage:.1f} nurse shortage"
            else:
                shortage_text = "Adequate staff"
                
            st.metric("Staff Readiness", f"{min(staff_ratio, 100):.0f}%", shortage_text, delta_color="inverse")
        else:
            st.metric("Staff Readiness", "78%", "-5% shortage", delta_color="inverse")

# MAIN PREDICTION CHART
st.subheader("📈 Bed Occupancy Forecast")

# Create chart with alert zones
if use_real_data and uploaded_file is not None and 'bed_occupancy' in df.columns:
    # Use real data for chart
    fig = px.line(
        df, 
        x='date' if 'date' in df.columns else df.index,
        y='bed_occupancy',
        title='Historical Bed Occupancy Trend',
        labels={'bed_occupancy': 'Occupancy Rate (%)', 'date': 'Date'}
    )
    
    # Add alert zones for real data too
    fig.add_hrect(y0=90, y1=100, fillcolor="red", opacity=0.2, 
                  annotation_text="CRITICAL ZONE", annotation_position="top left")
    fig.add_hrect(y0=80, y1=90, fillcolor="orange", opacity=0.2,
                  annotation_text="WARNING ZONE")
                  
else:
    # Use mock data for chart
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

# Calculate max occupancy for alerts
if use_real_data and uploaded_file is not None and 'bed_occupancy' in df.columns:
    max_occupancy = df['bed_occupancy'].max()
else:
    max_occupancy = df['occupancy_rate'].max() if 'occupancy_rate' in df.columns else 84

if max_occupancy >= 95:
    st.error("""
    🚨 **CRITICAL ALERT**: Capacity Crisis Detected
    - Maximum occupancy: 95%+
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
    ⚠️ **HIGH ALERT**: Resource Strain Detected  
    - Maximum occupancy: 85-94%
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
    - Maximum occupancy: <85%
    - Standard protocols sufficient
    """)

# ADDITIONAL ANALYSIS FOR REAL DATA
if use_real_data and uploaded_file is not None:
    st.subheader("📋 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Admissions vs Discharges**")
        if 'admissions' in df.columns and 'discharges' in df.columns:
            net_change = df['admissions'].sum() - df['discharges'].sum()
            st.metric("Net Patient Change", f"{net_change:+.0f}", "patients")
            
            # Simple trend
            avg_admissions = df['admissions'].mean()
            avg_discharges = df['discharges'].mean()
            st.write(f"Average daily admissions: {avg_admissions:.1f}")
            st.write(f"Average daily discharges: {avg_discharges:.1f}")
    
    with col2:
        st.write("**ICU Utilization**")
        if 'icu_patients' in df.columns:
            avg_icu = df['icu_patients'].mean()
            max_icu = df['icu_patients'].max()
            icu_utilization = (avg_icu / 20) * 100  # Assuming 20 ICU beds
            st.metric("Average ICU Utilization", f"{icu_utilization:.1f}%")
            st.write(f"Peak ICU patients: {max_icu}")

# FOOTER
st.markdown("---")
st.markdown("*HealthFlow - Empowering Malaysian Healthcare with AI Predictions*")

# DEBUG INFO (Optional - remove for final version)
with st.sidebar.expander("Debug Info"):
    st.write("DataFrame shape:", df.shape)
    st.write("DataFrame columns:", list(df.columns))
    st.write("Use real data:", use_real_data)
    st.write("File uploaded:", uploaded_file is not None)
