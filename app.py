import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Hospital Utilization Analysis",
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
    .advice-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
    }
    .critical-card {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'scenario' not in st.session_state:
    st.session_state.scenario = "Normal Operations"

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def main():
    st.markdown('<h1 class="main-header">🏥 Hospital Scenario Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar - only create widgets once
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload your hospital dataset", type=["csv"], key="file_uploader")
        
        if uploaded_file is not None:
            if st.session_state.df is None:
                st.session_state.df = load_data(uploaded_file)
                st.success("Dataset loaded successfully!")
        
        st.header("🎯 Analysis Scenarios")
        scenario = st.selectbox(
            "Select Scenario to Analyze",
            ["Normal Operations", "Weekend Surge", "Seasonal Outbreak", "Emergency Crisis"],
            key="scenario_selector"
        )
        
        # Update session state
        st.session_state.scenario = scenario
    
    # Main content area
    if st.session_state.df is not None:
        display_scenario_analysis(st.session_state.df, st.session_state.scenario)
    else:
        display_sample_interface()

def display_sample_interface():
    """Display sample interface before data upload"""
    st.info("👆 Please upload your hospital dataset (final_cleaned_dataset.csv) to begin analysis")
    
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
    
    # Calculate metrics for advice system
    metrics = calculate_metrics(filtered_df)
    
    # Key Metrics
    st.subheader("Key Performance Indicators")
    display_key_metrics(filtered_df, scenario)
    
    # Charts and Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        display_occupancy_analysis(filtered_df, scenario)
        display_staff_analysis(filtered_df, scenario)
    
    with col2:
        display_icu_analysis(filtered_df, scenario)
        display_comparison_analysis(df, filtered_df, scenario)
    
    # AI Recommendations Section
    display_ai_recommendations(scenario, metrics, filtered_df)
    
    # Detailed Data
    st.subheader("📋 Detailed Service Data")
    display_detailed_table(filtered_df)

def filter_data_by_scenario(df, scenario):
    """Filter data based on the selected scenario"""
    
    if scenario == "Normal Operations":
        return df[df['event'] == 'none']
    
    elif scenario == "Weekend Surge":
        weekend_weeks = df['week'].sample(frac=0.3, random_state=42).unique()
        return df[df['week'].isin(weekend_weeks)]
    
    elif scenario == "Seasonal Outbreak":
        return df[df['event'] == 'flu']
    
    elif scenario == "Emergency Crisis":
        crisis_events = ['strike', 'flu']
        high_demand = df['patients_request'] > df['patients_request'].quantile(0.75)
        return df[(df['event'].isin(crisis_events)) | high_demand]
    
    return df

def calculate_metrics(filtered_df):
    """Calculate metrics for advice system"""
    
    avg_occupancy = filtered_df['occupancy_rate'].mean() * 100
    
    shortage_map = {'low': 1, 'moderate': 2, 'high': 3}
    filtered_df['shortage_score'] = filtered_df['staff_shortage_level'].map(shortage_map)
    avg_shortage = filtered_df['shortage_score'].mean()
    
    icu_data = filtered_df[filtered_df['service'] == 'ICU']
    if not icu_data.empty:
        icu_data['demand_score'] = icu_data['ICU_demand_level'].map(shortage_map)
        avg_icu_demand = icu_data['demand_score'].mean()
    else:
        avg_icu_demand = 0
    
    total_requested = filtered_df['patients_request'].sum()
    total_refused = filtered_df['patients_refused'].sum()
    refusal_rate = (total_refused / total_requested * 100) if total_requested > 0 else 0
    
    return {
        'occupancy': avg_occupancy,
        'shortage': avg_shortage,
        'icu_demand': avg_icu_demand,
        'refusal_rate': refusal_rate
    }

def display_key_metrics(filtered_df, scenario):
    """Display key metrics for the scenario"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Bed Occupancy Rate
    avg_occupancy = filtered_df['occupancy_rate'].mean() * 100
    occupancy_status = "🟢 Normal" if avg_occupancy < 80 else "🟡 High" if avg_occupancy < 95 else "🔴 Critical"
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Bed Occupancy", f"{avg_occupancy:.1f}%", occupancy_status)
        st.progress(avg_occupancy / 100)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Staff Shortage Level
    shortage_mapping = {'low': 1, 'moderate': 2, 'high': 3}
    filtered_df['shortage_score'] = filtered_df['staff_shortage_level'].map(shortage_mapping)
    avg_shortage = filtered_df['shortage_score'].mean()
    shortage_status = "🟢 Low" if avg_shortage < 1.5 else "🟡 Moderate" if avg_shortage < 2.5 else "🔴 High"
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("👨‍⚕️ Staff Shortage Level", shortage_status, f"Score: {avg_shortage:.1f}/3")
        st.progress(avg_shortage / 3)
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
        demand_status = "No Data"
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💊 ICU Demand Level", demand_status, f"Score: {avg_demand:.1f}/3")
        st.progress(avg_demand / 3)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Patient Refusal Rate
    total_requested = filtered_df['patients_request'].sum()
    total_refused = filtered_df['patients_refused'].sum()
    refusal_rate = (total_refused / total_requested * 100) if total_requested > 0 else 0
    refusal_status = "🟢 Low" if refusal_rate < 10 else "🟡 Moderate" if refusal_rate < 25 else "🔴 High"
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("❌ Patient Refusal Rate", f"{refusal_rate:.1f}%", refusal_status)
        st.progress(min(refusal_rate / 50, 1.0))
        st.write(f"{total_refused:,} refused of {total_requested:,} requested")
        st.markdown('</div>', unsafe_allow_html=True)

def display_occupancy_analysis(filtered_df, scenario):
    """Display bed occupancy analysis"""
    
    st.subheader("🛏️ Bed Occupancy Analysis")
    
    occupancy_by_service = filtered_df.groupby('service')['occupancy_rate'].mean() * 100
    chart_data = pd.DataFrame({
        'Service': occupancy_by_service.index,
        'Occupancy Rate %': occupancy_by_service.values
    })
    
    st.bar_chart(chart_data.set_index('Service'))
    
    # Occupancy distribution
    st.write("**Occupancy Distribution:**")
    col1, col2, col3 = st.columns(3)
    
    low_occ = len(filtered_df[filtered_df['occupancy_rate'] < 0.8]) / len(filtered_df) * 100
    high_occ = len(filtered_df[(filtered_df['occupancy_rate'] >= 0.8) & (filtered_df['occupancy_rate'] < 0.95)]) / len(filtered_df) * 100
    critical_occ = len(filtered_df[filtered_df['occupancy_rate'] >= 0.95]) / len(filtered_df) * 100
    
    with col1:
        st.metric("Low (<80%)", f"{low_occ:.1f}%", delta_color="off")
    with col2:
        st.metric("High (80-95%)", f"{high_occ:.1f}%", delta_color="off")
    with col3:
        st.metric("Critical (≥95%)", f"{critical_occ:.1f}%", delta_color="off")
    
    # Occupancy status summary
    avg_occupancy = filtered_df['occupancy_rate'].mean() * 100
    if avg_occupancy < 80:
        st.success("✅ Bed occupancy is at manageable levels")
    elif avg_occupancy < 95:
        st.warning("⚠️ Bed occupancy is high - consider additional resources")
    else:
        st.error("🚨 Critical bed occupancy - immediate action required")

def display_staff_analysis(filtered_df, scenario):
    """Display staff shortage analysis"""
    
    st.subheader("Staff Shortage Analysis")
    
    shortage_dist = filtered_df['staff_shortage_level'].value_counts()
    shortage_data = pd.DataFrame({
        'Level': shortage_dist.index,
        'Count': shortage_dist.values
    })
    
    st.bar_chart(shortage_data.set_index('Level'))
    
    # Staff metrics
    col1, col2 = st.columns(2)
    
    with col1:
        avg_morale = filtered_df['staff_morale'].mean()
        morale_status = "Good" if avg_morale > 70 else "Fair" if avg_morale > 50 else "Poor"
        st.metric("Average Staff Morale", f"{avg_morale:.1f}", morale_status)
        
        st.write("Morale Level:")
        if avg_morale > 70:
            st.success("High morale 🎉")
        elif avg_morale > 50:
            st.warning("Moderate morale 😊")
        else:
            st.error("Low morale 😔")
    
    with col2:
        high_shortage_pct = (shortage_dist.get('high', 0) / len(filtered_df)) * 100
        st.metric("High Shortage Frequency", f"{high_shortage_pct:.1f}%")
        
        if high_shortage_pct > 30:
            st.error("Frequent staff shortages 🚨")
        elif high_shortage_pct > 15:
            st.warning("Occasional staff shortages ⚠️")
        else:
            st.success("Minimal staff shortages ✅")

def display_icu_analysis(filtered_df, scenario):
    """Display ICU demand analysis"""
    
    st.subheader("ICU Demand Analysis")
    
    icu_data = filtered_df[filtered_df['service'] == 'ICU']
    
    if icu_data.empty:
        st.warning("No ICU data available for this scenario")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        demand_counts = icu_data['ICU_demand_level'].value_counts()
        demand_data = pd.DataFrame({
            'Demand Level': demand_counts.index,
            'Count': demand_counts.values
        })
        
        st.bar_chart(demand_data.set_index('Demand Level'))
        most_common_demand = demand_counts.index[0]
        st.write(f"**Most frequent demand level:** {most_common_demand}")
    
    with col2:
        avg_icu_occupancy = icu_data['occupancy_rate'].mean() * 100
        icu_refusal_rate = (icu_data['patients_refused'].sum() / icu_data['patients_request'].sum() * 100) if icu_data['patients_request'].sum() > 0 else 0
        
        st.metric("ICU Average Occupancy", f"{avg_icu_occupancy:.1f}%")
        st.metric("ICU Refusal Rate", f"{icu_refusal_rate:.1f}%")
        st.metric("ICU Patients Admitted", f"{icu_data['patients_admitted'].sum():,}")
        
        st.write("**ICU Capacity Status:**")
        if avg_icu_occupancy > 90:
            st.error("🚨 Critical: ICU near full capacity")
        elif avg_icu_occupancy > 75:
            st.warning("⚠️ Warning: ICU capacity strained")
        else:
            st.success("✅ Stable: ICU capacity adequate")

def display_comparison_analysis(df, filtered_df, scenario):
    """Display comparison with baseline"""
    
    st.subheader("📈 Scenario vs Baseline Comparison")
    
    baseline_df = df[df['event'] == 'none']
    
    metrics = ['Bed Occupancy', 'Staff Morale', 'Refusal Rate']
    baseline_values = [
        baseline_df['occupancy_rate'].mean() * 100,
        baseline_df['staff_morale'].mean(),
        (baseline_df['patients_refused'].sum() / baseline_df['patients_request'].sum() * 100) if baseline_df['patients_request'].sum() > 0 else 0
    ]
    scenario_values = [
        filtered_df['occupancy_rate'].mean() * 100,
        filtered_df['staff_morale'].mean(),
        (filtered_df['patients_refused'].sum() / filtered_df['patients_request'].sum() * 100) if filtered_df['patients_request'].sum() > 0 else 0
    ]
    
    comparison_data = {
        'Metric': metrics,
        'Baseline': [f"{val:.1f}{'%' if i != 1 else ''}" for i, val in enumerate(baseline_values)],
        f'{scenario}': [f"{val:.1f}{'%' if i != 1 else ''}" for i, val in enumerate(scenario_values)],
        'Change': [f"{'📈' if scenario_values[i] > baseline_values[i] else '📉' if scenario_values[i] < baseline_values[i] else '➡️'} {abs(scenario_values[i] - baseline_values[i]):.1f}{'%' if i != 1 else ''}" 
                 for i in range(len(metrics))]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    # Summary insights
    st.subheader("💡 Key Insights")
    
    occupancy_change = scenario_values[0] - baseline_values[0]
    morale_change = scenario_values[1] - baseline_values[1]
    refusal_change = scenario_values[2] - baseline_values[2]
    
    if occupancy_change > 10:
        st.error(f"**Occupancy Impact:** {scenario} shows {occupancy_change:.1f}% higher bed occupancy than normal")
    elif occupancy_change > 5:
        st.warning(f"**Occupancy Impact:** {scenario} shows {occupancy_change:.1f}% higher bed occupancy than normal")
    else:
        st.success(f"**Occupancy Impact:** Bed occupancy is similar to normal operations")
    
    if morale_change < -10:
        st.error(f"**Staff Impact:** Staff morale is {abs(morale_change):.1f} points lower than normal")
    elif morale_change < -5:
        st.warning(f"**Staff Impact:** Staff morale is {abs(morale_change):.1f} points lower than normal")
    else:
        st.success(f"**Staff Impact:** Staff morale remains stable")

def display_ai_recommendations(scenario, metrics, filtered_df):
    """Display AI-powered recommendations"""
    
    st.markdown("---")
    st.subheader("Recommendations & Action Plan")
    
    advice_list = generate_advice(scenario, metrics, filtered_df)
    
    if not advice_list:
        st.info("🎯 **System Operating Optimally** - No immediate actions required")
        return
    
    # Categorize advice by priority
    critical_advice = [adv for adv in advice_list if "🔴" in adv[0]]
    warning_advice = [adv for adv in advice_list if "🟡" in adv[0]]
    normal_advice = [adv for adv in advice_list if "🟢" in adv[0]]
    
    # Display critical advice first
    for priority, advice_group in [("🚨 Critical Actions", critical_advice), 
                                 ("⚠️ Important Recommendations", warning_advice),
                                 ("💡 Optimization Opportunities", normal_advice)]:
        
        if advice_group:
            st.write(f"**{priority}**")
            
            for title, description in advice_group:
                if "🔴" in title:
                    st.markdown(f'<div class="critical-card"><h4>{title}</h4><p>{description}</p></div>', unsafe_allow_html=True)
                elif "🟡" in title:
                    st.markdown(f'<div class="warning-card"><h4>{title}</h4><p>{description}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="advice-card"><h4>{title}</h4><p>{description}</p></div>', unsafe_allow_html=True)

def generate_advice(scenario, metrics, filtered_df):
    """Generate AI-powered advice based on scenario and metrics"""
    
    advice = []
    occupancy = metrics['occupancy']
    shortage = metrics['shortage']
    icu_demand = metrics['icu_demand']
    refusal_rate = metrics['refusal_rate']
    
    # Scenario-specific advice
    if scenario == "Normal Operations":
        advice.extend(generate_normal_advice(occupancy, shortage, icu_demand, refusal_rate))
    elif scenario == "Weekend Surge":
        advice.extend(generate_weekend_advice(occupancy, shortage, icu_demand, refusal_rate))
    elif scenario == "Seasonal Outbreak":
        advice.extend(generate_seasonal_advice(occupancy, shortage, icu_demand, refusal_rate))
    elif scenario == "Emergency Crisis":
        advice.extend(generate_emergency_advice(occupancy, shortage, icu_demand, refusal_rate))
    
    # General advice
    advice.extend(generate_general_advice(occupancy, shortage, icu_demand, refusal_rate))
    
    return advice

def generate_normal_advice(occupancy, shortage, icu_demand, refusal_rate):
    advice = []
    if occupancy < 70:
        advice.append(("🟢 **Optimization Opportunity**", "Consider consolidating wards to improve efficiency"))
    if shortage > 2.0:
        advice.append(("🟡 **Staff Development**", "Invest in staff training and retention programs"))
    if refusal_rate < 5:
        advice.append(("🟢 **Capacity Planning**", "Excellent patient access. Consider expanding services"))
    return advice

def generate_weekend_advice(occupancy, shortage, icu_demand, refusal_rate):
    advice = []
    if occupancy > 85:
        advice.append(("🟡 **Weekend Staffing**", "Implement flexible weekend scheduling with premium pay"))
    if refusal_rate > 15:
        advice.append(("🟡 **Patient Flow**", "Create weekend-specific discharge protocols"))
    if shortage > 2.2:
        advice.append(("🟡 **Cross-Training**", "Cross-train staff for multiple roles"))
    return advice

def generate_seasonal_advice(occupancy, shortage, icu_demand, refusal_rate):
    advice = []
    if occupancy > 90:
        advice.append(("🟡 **Seasonal Preparedness**", "Pre-arrange temporary bed capacity"))
    if icu_demand > 2.5:
        advice.append(("🟡 **ICU Planning**", "Establish step-down units for ICU overflow"))
    if refusal_rate > 20:
        advice.append(("🟡 **Triage Optimization**", "Implement enhanced triage protocols"))
    return advice

def generate_emergency_advice(occupancy, shortage, icu_demand, refusal_rate):
    advice = []
    if occupancy > 95:
        advice.append(("🔴 **Crisis Response**", "Activate emergency overflow protocol"))
    if shortage > 2.8:
        advice.append(("🔴 **Staff Mobilization**", "Implement mandatory overtime immediately"))
    if icu_demand > 2.7:
        advice.append(("🔴 **ICU Expansion**", "Convert recovery areas to temporary ICU beds"))
    if refusal_rate > 30:
        advice.append(("🔴 **Patient Diversion**", "Coordinate with nearby hospitals for transfers"))
    return advice

def generate_general_advice(occupancy, shortage, icu_demand, refusal_rate):
    advice = []
    if occupancy > 95:
        advice.append(("🔴 **Critical Occupancy**", "Open all reserve beds, activate emergency staffing"))
    elif occupancy > 85:
        advice.append(("🟡 **High Occupancy**", "Prepare overflow areas, review discharge readiness"))
    
    if shortage > 2.5:
        advice.append(("🔴 **Severe Staff Shortage**", "Deploy administrative staff to clinical support"))
    elif shortage > 2.0:
        advice.append(("🟡 **Moderate Staff Shortage**", "Consider agency staff or float pool activation"))
    
    if icu_demand > 2.5:
        advice.append(("🔴 **High ICU Demand**", "Prioritize ICU admissions, expedite transfers"))
    
    if refusal_rate > 25:
        advice.append(("🔴 **Critical Refusal Rate**", "Implement disaster triage protocols"))
    elif refusal_rate > 15:
        advice.append(("🟡 **Elevated Refusal Rate**", "Optimize bed turnover and admission criteria"))
    
    return advice

def display_detailed_table(filtered_df):
    """Display detailed data table"""
    
    display_columns = ['week', 'month', 'service', 'available_beds', 'patients_request', 
                     'patients_admitted', 'patients_refused', 'occupancy_rate', 
                     'staff_shortage_level', 'ICU_demand_level', 'event']
    
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    detailed_df = filtered_df[available_columns]
    
    detailed_df['refusal_rate'] = (detailed_df['patients_refused'] / detailed_df['patients_request'] * 100).round(1)
    detailed_df['occupancy_rate_pct'] = (detailed_df['occupancy_rate'] * 100).round(1)
    
    st.dataframe(detailed_df, use_container_width=True, height=400)

if __name__ == "__main__":
    main()
