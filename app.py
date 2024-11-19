# app.py
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

# Data Processing Functions
def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the CSV data"""
    df = pd.read_csv(uploaded_file)
    
    # Clean column names - make them lowercase and remove whitespace
    df.columns = df.columns.str.strip().str.lower()
    
    # Map common column names to standardized names
    column_mapping = {
        'power': ['power', 'w', 'laser power'],
        'speed': ['speed', 'mm/s', 'scanning speed'],
        'hatch': ['hatch', 'hatch spacing', 'hatch distance'],
        'thickness': ['thickness', 'layer thickness', 'layer height'],
        'uts': ['uts', 'ultimate tensile strength', 'tensile strength'],
        'ys': ['ys', 'yield strength'],
        'elongation': ['elongation', 'elongation %', '%'],
        'hardness': ['hardness', 'hv', 'hardness (hv)'],
        'solution temp': ['solution temp', 'solution temperature'],
        'sol time': ['sol time', 'solution time'],
        'ageing temp': ['ageing temp', 'aging temperature'],
        'ageing time': ['ageing time', 'aging time']
    }

    # Standardize column names
    new_columns = {}
    for standard_name, variations in column_mapping.items():
        for col in df.columns:
            if col in variations:
                new_columns[col] = standard_name
    
    df = df.rename(columns=new_columns)
    
    # Handle missing values for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Calculate energy density if possible
    required_columns = ['power', 'speed', 'hatch', 'thickness']
    if all(col in df.columns for col in required_columns):
        df['energy_density'] = df['power'] / (df['speed'] * df['hatch'] * df['thickness'])
    else:
        st.warning("Cannot calculate energy density - missing required parameters")
        df['energy_density'] = np.nan
    
    return df

def extract_features(df):
    """Extract relevant features for analysis"""
    process_params = ['power', 'speed', 'hatch', 'thickness', 'energy_density']
    ht_params = ['solution temp', 'sol time', 'ageing temp', 'ageing time']
    mech_props = ['uts', 'ys', 'elongation', 'hardness']
    
    # Only include columns that exist in the dataframe
    process_params = [col for col in process_params if col in df.columns]
    ht_params = [col for col in ht_params if col in df.columns]
    mech_props = [col for col in mech_props if col in df.columns]
    
    return {
        'process': df[process_params] if process_params else pd.DataFrame(),
        'heat_treatment': df[ht_params] if ht_params else pd.DataFrame(),
        'properties': df[mech_props] if mech_props else pd.DataFrame()
    }

[... rest of the code remains the same until the visualization functions ...]

def create_process_property_plot(df, x_param, y_param, color_by='direction'):
    """Create scatter plot of process-property relationships"""
    if x_param not in df.columns or y_param not in df.columns:
        st.error(f"Required columns {x_param} and/or {y_param} not found in data")
        return None
        
    fig = px.scatter(df, 
                    x=x_param,
                    y=y_param,
                    color=color_by if color_by in df.columns else None,
                    title=f"{y_param.upper()} vs {x_param.upper()}",
                    labels={x_param: x_param.upper(), 
                           y_param: y_param.upper()})
    return fig

def create_parallel_coordinates_plot(df):
    """Create parallel coordinates plot for heat treatment analysis"""
    dimensions = ['solution temp', 'sol time', 'ageing temp', 
                 'ageing time', 'uts', 'ys', 'elongation']
    
    # Only use dimensions that exist in the dataframe
    available_dims = [dim for dim in dimensions if dim in df.columns]
    
    if not available_dims:
        st.error("No required columns found for parallel coordinates plot")
        return None
        
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=df[available_dims[0]],
                     colorscale='Viridis'),
            dimensions=[dict(range=[df[dim].min(), df[dim].max()],
                           label=dim.upper(),
                           values=df[dim]) for dim in available_dims]
        )
    )
    return fig

[... rest of the code remains the same ...]

# Main Application
def main():
    st.title("LPBF AlSi10Mg Analysis Dashboard")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Analysis",
        ["Data Upload & Overview", 
         "Process-Property Analysis",
         "Heat Treatment Analysis",
         "Statistical Analysis",
         "Machine Learning"]
    )
    
    # Data Upload
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    uploaded_file = st.file_uploader("Upload LPBF data (CSV)", type='csv')
    if uploaded_file is not None:
        try:
            st.session_state.data = load_and_preprocess_data(uploaded_file)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.data = None
            
    [... rest of the main() function remains the same ...]

if __name__ == "__main__":
    main()
