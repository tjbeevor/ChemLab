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
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        df.columns = df.columns.str.replace(' ', '_')
        
        # Map common column names
        column_map = {
            col: 'power' for col in df.columns if any(x in col.lower() for x in ['power', '_w_'])
        }
        column_map.update({
            col: 'speed' for col in df.columns if any(x in col.lower() for x in ['speed', 'mm/s'])
        })
        column_map.update({
            col: 'hatch' for col in df.columns if 'hatch' in col.lower()
        })
        column_map.update({
            col: 'thickness' for col in df.columns if 'thickness' in col.lower()
        })
        column_map.update({
            col: 'uts' for col in df.columns if any(x in col.lower() for x in ['uts', 'tensile_strength'])
        })
        column_map.update({
            col: 'ys' for col in df.columns if any(x in col.lower() for x in ['ys', 'yield_strength'])
        })
        column_map.update({
            col: 'elongation' for col in df.columns if 'elongation' in col.lower()
        })
        column_map.update({
            col: 'hardness' for col in df.columns if any(x in col.lower() for x in ['hardness', 'hv'])
        })
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Convert numeric columns safely
        for col in df.columns:
            try:
                if 'direction' not in col.lower():
                    df[col] = pd.to_numeric(df[col])
            except:
                continue
        
        # Handle direction column
        direction_cols = [col for col in df.columns if 'direction' in col.lower()]
        if direction_cols:
            df['direction'] = df[direction_cols[0]]
        
        # Calculate energy density
        if all(x in df.columns for x in ['power', 'speed', 'hatch', 'thickness']):
            mask = (df['speed'] != 0) & (df['hatch'] != 0) & (df['thickness'] != 0)
            df.loc[mask, 'energy_density'] = df.loc[mask, 'power'] / (df.loc[mask, 'speed'] * df.loc[mask, 'hatch'] * df.loc[mask, 'thickness'])
        
        return df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def create_process_property_plot(df, x_param, y_param, color_by='direction'):
    """Create scatter plot"""
    if x_param not in df.columns or y_param not in df.columns:
        return None
        
    fig = px.scatter(
        df, 
        x=x_param,
        y=y_param,
        color=color_by if color_by in df.columns else None,
        title=f"{y_param.upper()} vs {x_param.upper()}",
        labels={x_param: x_param.upper(), y_param: y_param.upper()}
    )
    
    fig.update_layout(height=600)
    return fig

def create_parallel_coordinates_plot(df):
    """Create parallel coordinates plot"""
    ht_cols = ['solution_temp', 'ageing_temp', 'uts', 'ys', 'elongation', 'hardness']
    available_cols = [col for col in ht_cols if col in df.columns]
    
    if not available_cols:
        return None
        
    # Convert columns to numeric and handle NaN values
    plot_df = df[available_cols].copy()
    for col in available_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    
    plot_df = plot_df.fillna(plot_df.mean())
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=plot_df[available_cols[0]], colorscale='Viridis'),
            dimensions=[dict(range=[plot_df[dim].min(), plot_df[dim].max()],
                           label=dim.upper(),
                           values=plot_df[dim]) for dim in available_cols]
        )
    )
    
    fig.update_layout(height=600)
    return fig

def main():
    st.title("LPBF AlSi10Mg Analysis Dashboard")
    
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
                st.subheader("Data Overview")
                st.write("Dataset Shape:", df.shape)
                st.write(df.head())
                
                st.subheader("Statistical Summary")
                st.write(df.describe())
                
            elif page == "Process-Property Analysis":
                st.subheader("Process-Property Relationships")
                
                # Define process parameters and properties
                process_params = ['power', 'speed', 'hatch', 'thickness', 'energy_density']
                properties = ['uts', 'ys', 'elongation', 'hardness']
                
                # Filter available columns
                available_process = [p for p in process_params if p in df.columns]
                available_properties = [p for p in properties if p in df.columns]
                
                if not available_process or not available_properties:
                    st.error("No process parameters or properties found in the data")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_param = st.selectbox("X-axis Parameter", options=available_process)
                    with col2:
                        y_param = st.selectbox("Y-axis Parameter", options=available_properties)
                    
                    if x_param and y_param:
                        fig = create_process_property_plot(df, x_param, y_param)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            elif page == "Heat Treatment Analysis":
                st.subheader("Heat Treatment Analysis")
                
                fig = create_parallel_coordinates_plot(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data for heat treatment analysis")

if __name__ == "__main__":
    main()
