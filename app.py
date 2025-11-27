import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Hospital Scenario Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .scenario-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🏥 Hospital Scenario Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # File upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your hospital dataset", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.success("✅ Dataset loaded successfully!")
        
        # Main analysis
        st.sidebar.header("🎯 Analysis Scenarios")
        scenario = st.sidebar.selectbox(
            "Select Scenario to Analyze",
            ["Normal Operations", "Weekend Surge", "Seasonal Outbreak", "Emergency Crisis"]
        )
        
        display_scenario_analysis(df, scenario)
        
    else:
        st.info("👆 Please upload your hospital dataset (final_cleaned_dataset.csv) to begin analysis")
        display_sample_interface()

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def display_sample_interface():
    """Display sample interface before data upload"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
        st.subheader("📊 Normal Operations")
        st.write("• Baseline occupancy rates")
        st.write("• Regular staff levels")
        st.write("• Standard ICU demand")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
        st.subheader("📈 Weekend Surge")
        st.write("• Increased patient volume")
        st.write("• Reduced staff availability")
        st.write("• Higher refusal rates")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
        st.subheader("🦠 Seasonal Outbreak")
        st.write("• Flu season impact")
        st.write("• Staff shortage peaks")
        st.write("• ICU capacity strain")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
        st.subheader("🚨 Emergency Crisis")
        st.write("• Maximum occupancy")
        st.write("• Critical staff shortage")
        st.write("• ICU demand surge")
        st.markdown('</div>', unsafe_allow_html=True)

def display_scenario_analysis(df, scenario):
    """Display analysis for the selected scenario"""
    
    st.markdown(f'<h2 style="text-align: center; color: #1f77b4;">{scenario} Analysis</h2>', unsafe_allow_html=True)
    
    # Filter data based on scenario
    filtered_df = filter_data_by_scenario(df, scenario)
    
    # Key Metrics
    st.subheader("📊 Key Performance Indicators")
    display_key_metrics(filtered_df, scenario)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        display_occupancy_analysis(filtered_df, scenario)
        display_staff_analysis(filtered_df, scenario)
    
    with col2:
        display_icu_analysis(filtered_df, scenario)
        display_comparison_chart(df, filtered_df, scenario)
    
    # Detailed Data
    st.subheader("📋 Detailed Service Data")
    display_detailed_table(filtered_df)

def filter_data_by_scenario(df, scenario):
    """Filter data based on the selected scenario"""
    
    if scenario == "Normal Operations":
        # Normal conditions - exclude events and extreme values
        return df[df['event'] == 'none']
    
    elif scenario == "Weekend Surge":
        # Weekend-like conditions - higher patient volume, some staff issues
        weekend_weeks = df['week'].sample(frac=0.3, random_state=42).unique()
        return df[df['week'].isin(weekend_weeks)]
    
    elif scenario == "Seasonal Outbreak":
        # Flu season conditions
        return df[df['event'] == 'flu']
    
    elif scenario == "Emergency Crisis":
        # Crisis conditions - strikes, high demand
        crisis_events = ['strike', 'flu']  # Combine multiple crisis indicators
        high_demand = df['patients_request'] > df['patients_request'].quantile(0.75)
        return df[(df['event'].isin(crisis_events)) | high_demand]
    
    return df

def display_key_metrics(filtered_df, scenario):
    """Display key metrics for the scenario"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Bed Occupancy Rate
    avg_occupancy = filtered_df['occupancy_rate'].mean() * 100
    occupancy_trend = "🟢 Normal" if avg_occupancy < 80 else "🟡 High" if avg_occupancy < 95 else "🔴 Critical"
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "🛏️ Average Bed Occupancy Rate", 
            f"{avg_occupancy:.1f}%",
            occupancy_trend
        )
        st.write(f"Scenario: {scenario}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Staff Shortage Level
    shortage_mapping = {'low': 1, 'moderate': 2, 'high': 3}
    filtered_df['shortage_score'] = filtered_df['staff_shortage_level'].map(shortage_mapping)
    avg_shortage = filtered_df['shortage_score'].mean()
    shortage_status = "🟢 Low" if avg_shortage < 1.5 else "🟡 Moderate" if avg_shortage < 2.5 else "🔴 High"
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "👨‍⚕️ Staff Shortage Level",
            shortage_status,
            f"Score: {avg_shortage:.1f}"
        )
        st.write("Based on shortage levels")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ICU Demand Level
    icu_data = filtered_df[filtered_df['service'] == 'ICU']
    if not icu_data.empty:
        demand_mapping = {'low': 1, 'moderate': 2, 'high': 3}
        icu_data['demand_score'] = icu_data['ICU_demand_level'].map(demand_mapping)
        avg_demand = icu_data['demand_score'].mean()
        demand_status = "🟢 Low" if avg_demand < 1.5 else "🟡 Moderate" if avg_demand < 2.5 else "🔴 High"
    else:
        avg_demand = 0
        demand_status = "No ICU Data"
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "💊 ICU Demand Level",
            demand_status,
            f"Score: {avg_demand:.1f}"
        )
        st.write("ICU-specific analysis")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Patient Refusal Rate
    total_requested = filtered_df['patients_request'].sum()
    total_refused = filtered_df['patients_refused'].sum()
    refusal_rate = (total_refused / total_requested * 100) if total_requested > 0 else 0
    refusal_status = "🟢 Low" if refusal_rate < 10 else "🟡 Moderate" if refusal_rate < 25 else "🔴 High"
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "❌ Patient Refusal Rate",
            f"{refusal_rate:.1f}%",
            refusal_status
        )
        st.write(f"{total_refused:,} refused of {total_requested:,} requested")
        st.markdown('</div>', unsafe_allow_html=True)

def display_occupancy_analysis(filtered_df, scenario):
    """Display bed occupancy analysis"""
    
    st.subheader("🛏️ Bed Occupancy Analysis")
    
    # Occupancy by service
    fig, ax = plt.subplots(figsize=(10, 6))
    
    occupancy_by_service = filtered_df.groupby('service')['occupancy_rate'].mean() * 100
    
    colors = ['#2E86AB' if rate < 80 else '#F7B801' if rate < 95 else '#FF6B6B' 
             for rate in occupancy_by_service.values]
    
    bars = ax.bar(occupancy_by_service.index, occupancy_by_service.values, color=colors, alpha=0.8)
    ax.set_title(f'Average Occupancy Rate by Service - {scenario}')
    ax.set_ylabel('Occupancy Rate (%)')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add threshold lines
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='High Occupancy (80%)')
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Critical (95%)')
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Occupancy distribution
    st.write("**Occupancy Distribution:**")
    col1, col2, col3 = st.columns(3)
    
    low_occ = len(filtered_df[filtered_df['occupancy_rate'] < 0.8]) / len(filtered_df) * 100
    high_occ = len(filtered_df[(filtered_df['occupancy_rate'] >= 0.8) & (filtered_df['occupancy_rate'] < 0.95)]) / len(filtered_df) * 100
    critical_occ = len(filtered_df[filtered_df['occupancy_rate'] >= 0.95]) / len(filtered_df) * 100
    
    with col1:
        st.metric("Low (<80%)", f"{low_occ:.1f}%")
    with col2:
        st.metric("High (80-95%)", f"{high_occ:.1f}%")
    with col3:
        st.metric("Critical (≥95%)", f"{critical_occ:.1f}%")

def display_staff_analysis(filtered_df, scenario):
    """Display staff shortage analysis"""
    
    st.subheader("👨‍⚕️ Staff Shortage Analysis")
    
    # Staff shortage by service
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Shortage levels by service
    shortage_by_service = pd.crosstab(filtered_df['service'], filtered_df['staff_shortage_level'])
    shortage_by_service.plot(kind='bar', ax=ax1, color=['#2E86AB', '#F7B801', '#FF6B6B'])
    ax1.set_title(f'Staff Shortage Levels by Service - {scenario}')
    ax1.set_ylabel('Number of Records')
    ax1.legend(title='Shortage Level')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Staff morale vs shortage
    shortage_mapping = {'low': 1, 'moderate': 2, 'high': 3}
    filtered_df['shortage_numeric'] = filtered_df['staff_shortage_level'].map(shortage_mapping)
    
    scatter = ax2.scatter(filtered_df['shortage_numeric'], filtered_df['staff_morale'], 
                        alpha=0.6, c=filtered_df['shortage_numeric'], cmap='RdYlGn_r')
    ax2.set_xlabel('Staff Shortage Level (1=Low, 3=High)')
    ax2.set_ylabel('Staff Morale')
    ax2.set_title('Staff Morale vs Shortage Level')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Staff metrics summary
    st.write("**Staff Metrics Summary:**")
    col1, col2 = st.columns(2)
    
    with col1:
        avg_morale = filtered_df['staff_morale'].mean()
        morale_status = "🟢 Good" if avg_morale > 70 else "🟡 Fair" if avg_morale > 50 else "🔴 Poor"
        st.metric("Average Staff Morale", f"{avg_morale:.1f}", morale_status)
    
    with col2:
        shortage_dist = filtered_df['staff_shortage_level'].value_counts(normalize=True) * 100
        high_shortage = shortage_dist.get('high', 0)
        st.metric("High Shortage Occurrence", f"{high_shortage:.1f}%")

def display_icu_analysis(filtered_df, scenario):
    """Display ICU demand analysis"""
    
    st.subheader("💊 ICU Demand Analysis")
    
    icu_data = filtered_df[filtered_df['service'] == 'ICU']
    
    if icu_data.empty:
        st.warning("No ICU data available for this scenario")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ICU demand levels
        fig, ax = plt.subplots(figsize=(8, 6))
        demand_counts = icu_data['ICU_demand_level'].value_counts()
        
        colors = {'low': '#2E86AB', 'moderate': '#F7B801', 'high': '#FF6B6B'}
        color_list = [colors.get(level, '#999999') for level in demand_counts.index]
        
        wedges, texts, autotexts = ax.pie(demand_counts.values, labels=demand_counts.index, 
                                        autopct='%1.1f%%', colors=color_list, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(f'ICU Demand Level Distribution - {scenario}')
        st.pyplot(fig)
    
    with col2:
        # ICU occupancy and metrics
        avg_icu_occupancy = icu_data['occupancy_rate'].mean() * 100
        icu_refusal_rate = (icu_data['patients_refused'].sum() / icu_data['patients_request'].sum() * 100) if icu_data['patients_request'].sum() > 0 else 0
        
        st.metric("ICU Average Occupancy", f"{avg_icu_occupancy:.1f}%")
        st.metric("ICU Refusal Rate", f"{icu_refusal_rate:.1f}%")
        st.metric("ICU Patients Admitted", f"{icu_data['patients_admitted'].sum():,}")
        
        # ICU demand trends
        st.write("**ICU Capacity Status:**")
        if avg_icu_occupancy > 90:
            st.error("🚨 Critical: ICU near full capacity")
        elif avg_icu_occupancy > 75:
            st.warning("⚠️ Warning: ICU capacity strained")
        else:
            st.success("✅ Stable: ICU capacity adequate")

def display_comparison_chart(df, filtered_df, scenario):
    """Display comparison with baseline"""
    
    st.subheader("📈 Scenario vs Baseline Comparison")
    
    # Calculate baseline (normal operations)
    baseline_df = df[df['event'] == 'none']
    
    comparison_data = {
        'Metric': ['Bed Occupancy', 'Staff Morale', 'Refusal Rate', 'ICU Demand'],
        'Baseline': [
            baseline_df['occupancy_rate'].mean() * 100,
            baseline_df['staff_morale'].mean(),
            (baseline_df['patients_refused'].sum() / baseline_df['patients_request'].sum() * 100) if baseline_df['patients_request'].sum() > 0 else 0,
            baseline_df[baseline_df['service'] == 'ICU']['ICU_demand_level'].map({'low': 1, 'moderate': 2, 'high': 3}).mean() if not baseline_df[baseline_df['service'] == 'ICU'].empty else 0
        ],
        'Scenario': [
            filtered_df['occupancy_rate'].mean() * 100,
            filtered_df['staff_morale'].mean(),
            (filtered_df['patients_refused'].sum() / filtered_df['patients_request'].sum() * 100) if filtered_df['patients_request'].sum() > 0 else 0,
            filtered_df[filtered_df['service'] == 'ICU']['ICU_demand_level'].map({'low': 1, 'moderate': 2, 'high': 3}).mean() if not filtered_df[filtered_df['service'] == 'ICU'].empty else 0
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Baseline'], width, label='Baseline (Normal)', alpha=0.7)
    ax.bar(x + width/2, comparison_df['Scenario'], width, label=f'Scenario ({scenario})', alpha=0.7)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title(f'Scenario Comparison: {scenario} vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Metric'])
    ax.legend()
    
    # Add value labels on bars
    for i, (baseline, scenario_val) in enumerate(zip(comparison_df['Baseline'], comparison_df['Scenario'])):
        ax.text(i - width/2, baseline + 1, f'{baseline:.1f}', ha='center', va='bottom')
        ax.text(i + width/2, scenario_val + 1, f'{scenario_val:.1f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def display_detailed_table(filtered_df):
    """Display detailed data table"""
    
    # Select relevant columns for display
    display_columns = ['week', 'month', 'service', 'available_beds', 'patients_request', 
                     'patients_admitted', 'patients_refused', 'occupancy_rate', 
                     'staff_shortage_level', 'ICU_demand_level', 'event']
    
    # Filter to only include available columns
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    detailed_df = filtered_df[available_columns]
    
    # Add calculated columns
    detailed_df['refusal_rate'] = (detailed_df['patients_refused'] / detailed_df['patients_request'] * 100).round(1)
    
    st.dataframe(
        detailed_df.style.format({
            'occupancy_rate': '{:.1%}',
            'refusal_rate': '{:.1f}%'
        }).background_gradient(subset=['occupancy_rate', 'refusal_rate'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=400
    )

if __name__ == "__main__":
    main()
