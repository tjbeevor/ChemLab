import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Configure the page
st.set_page_config(page_title="LPBF AlSi10Mg Analysis", layout="wide")

def load_and_preprocess_data(file_content):
    """Load and preprocess the CSV data"""
    try:
        # Read CSV file
        df = pd.read_csv(file_content)
        
        # Debug column names
        st.write("Original columns:", df.columns.tolist())
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        df.columns = df.columns.str.replace(' ', '_')
        
        # Map specific material science column names
        process_params = {
            'power': ['power', 'w', 'laser_power'],
            'speed': ['speed', 'mm/s', 'scanning_speed', 'scan_speed'],
            'hatch': ['hatch', 'hatch_spacing', 'hatch_distance'],
            'thickness': ['thickness', 'layer_thickness'],
            'energy_density': ['energy_density', 'ved', 'volumetric_energy_density'],
            'p/v': ['p/v', 'power_speed_ratio'],
            'bt': ['bt', 'bed_temperature', 'build_temperature']
        }
        
        mechanical_props = {
            'uts': ['uts', 'tensile_strength', 'ultimate_tensile_strength'],
            'ys': ['ys', 'yield_strength', 'yield_stress'],
            'elongation': ['elongation', 'elongation_%', 'strain_at_break'],
            'hardness': ['hardness', 'hv', 'vickers_hardness']
        }
        
        heat_treatment = {
            'solution_temp': ['solution_temp', 'solution_temperature', 'heat_treatment_method_solution_temp'],
            'ageing_temp': ['ageing_temp', 'aging_temperature', 'heat_treatment_method_ageing_temp'],
            'solution_time': ['solution_time', 'sol_time', 'heat_treatment_method_sol_time'],
            'ageing_time': ['ageing_time', 'aging_time', 'heat_treatment_method_ageing_time']
        }
        
        # Function to find matching column
        def find_matching_column(columns, keywords):
            for col in columns:
                if any(keyword in col.lower() for keyword in keywords):
                    return col
            return None
        
        # Create column mapping
        column_map = {}
        for param, keywords in {**process_params, **mechanical_props, **heat_treatment}.items():
            matching_col = find_matching_column(df.columns, keywords)
            if matching_col:
                column_map[matching_col] = param
        
        # Debug mapped columns
        st.write("Mapped columns:", column_map)
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Convert numeric columns safely
        for col in df.columns:
            if col not in ['direction', 'scanning', 'machine', 'powder_mfg']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    continue
        
        # Handle direction column specifically
        direction_cols = [col for col in df.columns if 'direction' in col.lower()]
        if direction_cols:
            df['build_direction'] = df[direction_cols[0]]
        
        # Calculate energy density if not present
        if 'energy_density' not in df.columns and all(x in df.columns for x in ['power', 'speed', 'hatch', 'thickness']):
            mask = (df['speed'] > 0) & (df['hatch'] > 0) & (df['thickness'] > 0)
            df.loc[mask, 'energy_density'] = df.loc[mask, 'power'] / (df.loc[mask, 'speed'] * df.loc[mask, 'hatch'] * df.loc[mask, 'thickness'])
        
        # Debug available columns after processing
        st.write("Available process parameters:", [col for col in process_params.keys() if col in df.columns])
        st.write("Available mechanical properties:", [col for col in mechanical_props.keys() if col in df.columns])
        st.write("Available heat treatment parameters:", [col for col in heat_treatment.keys() if col in df.columns])
        
        return df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def create_process_property_plot(df, x_param, y_param):
    """Create scatter plot with build direction coloring"""
    if x_param not in df.columns or y_param not in df.columns:
        st.error(f"Required columns not found: {x_param} or {y_param}")
        return None
    
    # Create figure
    fig = px.scatter(
        df, 
        x=x_param,
        y=y_param,
        color='build_direction' if 'build_direction' in df.columns else None,
        title=f"Relationship between {y_param.upper()} and {x_param.upper()}",
        labels={
            x_param: x_param.upper(),
            y_param: y_param.upper(),
            'build_direction': 'Build Direction'
        }
    )
    
    # Add trendline
    fig.add_traces(
        px.scatter(
            df, 
            x=x_param, 
            y=y_param, 
            trendline="ols"
        ).data
    )
    
    fig.update_layout(
        height=600,
        xaxis_title=f"{x_param.upper()} ({get_unit(x_param)})",
        yaxis_title=f"{y_param.upper()} ({get_unit(y_param)})"
    )
    
    return fig

def get_unit(param):
    """Get units for parameters"""
    units = {
        'power': 'W',
        'speed': 'mm/s',
        'hatch': 'mm',
        'thickness': 'mm',
        'energy_density': 'J/mm³',
        'uts': 'MPa',
        'ys': 'MPa',
        'elongation': '%',
        'hardness': 'HV',
        'solution_temp': '°C',
        'ageing_temp': '°C',
        'solution_time': 'hrs',
        'ageing_time': 'hrs'
    }
    return units.get(param, '')

def create_parallel_coordinates_plot(df):
    """Create parallel coordinates plot for heat treatment analysis"""
    # Identify relevant columns
    ht_cols = ['solution_temp', 'ageing_temp', 'solution_time', 'ageing_time', 
               'uts', 'ys', 'elongation', 'hardness']
    available_cols = [col for col in ht_cols if col in df.columns]
    
    if len(available_cols) < 2:
        st.error("Insufficient heat treatment or mechanical property data for analysis")
        return None
    
    # Prepare data
    plot_df = df[available_cols].copy()
    
    # Handle missing values
    plot_df = plot_df.fillna(plot_df.mean())
    
    # Create parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=plot_df[available_cols[0]],
                colorscale='Viridis',
                showscale=True
            ),
            dimensions=[
                dict(
                    range=[plot_df[dim].min(), plot_df[dim].max()],
                    label=f"{dim.upper()} ({get_unit(dim)})",
                    values=plot_df[dim]
                ) for dim in available_cols
            ]
        )
    )
    
    fig.update_layout(
        height=600,
        title="Heat Treatment Parameters and Mechanical Properties Relationship"
    )
    
    return fig

def main():
    st.title("🔬 LPBF AlSi10Mg Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes Laser Powder Bed Fusion (LPBF) process parameters and their effects 
    on AlSi10Mg mechanical properties. Upload your data to explore process-property relationships 
    and heat treatment effects.
    """)
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Analysis",
        ["Data Upload & Overview", 
         "Process-Property Analysis",
         "Heat Treatment Analysis"]
    )
    
    # File upload
    uploaded_file = st.file_uploader("Upload LPBF data (CSV)", type='csv')
    
    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)
        
        if df is not None:
            if page == "Data Upload & Overview":
                st.subheader("📊 Data Overview")
                st.write("Dataset Shape:", df.shape)
                st.write(df.head())
                
                st.subheader("📈 Statistical Summary")
                st.write(df.describe())
                
            elif page == "Process-Property Analysis":
                st.subheader("🔍 Process-Property Relationships")
                
                # Define available parameters
                process_params = ['power', 'speed', 'hatch', 'thickness', 'energy_density', 'p/v', 'bt']
                properties = ['uts', 'ys', 'elongation', 'hardness']
                
                # Filter available columns
                available_process = [p for p in process_params if p in df.columns]
                available_properties = [p for p in properties if p in df.columns]
                
                if not available_process or not available_properties:
                    st.error("""
                    No process parameters or properties found in the data. 
                    Required columns should include parameters like 'power', 'speed', 'hatch' 
                    and properties like 'UTS', 'YS', 'elongation', 'hardness'.
                    """)
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_param = st.selectbox(
                            "Process Parameter (X-axis)", 
                            options=available_process,
                            format_func=lambda x: f"{x.upper()} ({get_unit(x)})"
                        )
                    with col2:
                        y_param = st.selectbox(
                            "Mechanical Property (Y-axis)", 
                            options=available_properties,
                            format_func=lambda x: f"{x.upper()} ({get_unit(x)})"
                        )
                    
                    if x_param and y_param:
                        fig = create_process_property_plot(df, x_param, y_param)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add statistical analysis
                            st.subheader("📊 Statistical Analysis")
                            correlation = df[x_param].corr(df[y_param])
                            st.write(f"Correlation coefficient between {x_param.upper()} and {y_param.upper()}: {correlation:.3f}")
            
            elif page == "Heat Treatment Analysis":
                st.subheader("🔥 Heat Treatment Analysis")
                
                fig = create_parallel_coordinates_plot(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""
                    The parallel coordinates plot shows relationships between heat treatment parameters 
                    and resulting mechanical properties. Each line represents a sample, and the color 
                    gradient helps identify patterns in the data.
                    """)

if __name__ == "__main__":
    main()
