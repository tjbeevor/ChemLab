import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="LPBF AlSi10Mg Analysis", layout="wide")

def clean_column_name(col):
    """Clean and standardize column names"""
    col = col.lower().strip()
    if 'uts' in col or 'tensile_strength' in col:
        return 'uts'
    elif 'ys' in col or 'yield' in col:
        return 'ys'
    elif 'elongation' in col:
        return 'elongation'
    elif 'hardness' in col or 'hv' in col:
        return 'hardness'
    elif 'power' in col:
        return 'power'
    elif 'speed' in col or 'mm/s' in col:
        return 'speed'
    elif 'hatch' in col:
        return 'hatch'
    elif 'thickness' in col:
        return 'thickness'
    elif 'direction' in col:
        return 'direction'
    elif 'solution_temp' in col:
        return 'solution_temp'
    elif 'ageing_temp' in col:
        return 'ageing_temp'
    return col

def load_data(uploaded_file):
    """Load and preprocess data"""
    try:
        # Read the file
        df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Convert numeric columns
        numeric_cols = ['power', 'speed', 'hatch', 'thickness', 'uts', 'ys', 
                       'elongation', 'hardness', 'solution_temp', 'ageing_temp']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate energy density if possible
        if all(x in df.columns for x in ['power', 'speed', 'hatch', 'thickness']):
            mask = (df['speed'] > 0) & (df['hatch'] > 0) & (df['thickness'] > 0)
            df.loc[mask, 'energy_density'] = df.loc[mask, 'power'] / (df.loc[mask, 'speed'] * df.loc[mask, 'hatch'] * df.loc[mask, 'thickness'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.title("LPBF AlSi10Mg Analysis Dashboard")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your LPBF data (CSV)", type='csv')
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # Show column names found
            st.write("Found columns:", list(df.columns))
            
            # Analysis selection
            analysis = st.radio(
                "Select Analysis Type",
                ["Process Parameters vs Properties", "Heat Treatment Analysis"]
            )
            
            if analysis == "Process Parameters vs Properties":
                # Process parameters analysis
                process_params = [col for col in ['power', 'speed', 'hatch', 'thickness', 'energy_density'] 
                                if col in df.columns]
                properties = [col for col in ['uts', 'ys', 'elongation', 'hardness'] 
                            if col in df.columns]
                
                if process_params and properties:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_param = st.selectbox("Process Parameter", process_params)
                    with col2:
                        y_param = st.selectbox("Mechanical Property", properties)
                    
                    if x_param and y_param:
                        fig = px.scatter(
                            df,
                            x=x_param,
                            y=y_param,
                            color='direction' if 'direction' in df.columns else None,
                            trendline="ols"
                        )
                        st.plotly_chart(fig)
                        
                        # Show correlation
                        corr = df[x_param].corr(df[y_param])
                        st.write(f"Correlation coefficient: {corr:.3f}")
                else:
                    st.error("No process parameters or mechanical properties found in the data")
            
            else:
                # Heat treatment analysis
                ht_params = [col for col in ['solution_temp', 'ageing_temp'] 
                           if col in df.columns]
                mech_props = [col for col in ['uts', 'ys', 'elongation', 'hardness'] 
                            if col in df.columns]
                
                if ht_params and mech_props:
                    plot_cols = ht_params + mech_props
                    plot_df = df[plot_cols].dropna()
                    
                    if not plot_df.empty:
                        fig = go.Figure(data=
                            go.Parcoords(
                                line=dict(color=plot_df[mech_props[0]], colorscale='Viridis'),
                                dimensions=[dict(range=[plot_df[col].min(), plot_df[col].max()],
                                               label=col.upper(),
                                               values=plot_df[col]) for col in plot_cols]
                            )
                        )
                        st.plotly_chart(fig)
                    else:
                        st.error("Insufficient data for heat treatment analysis")
                else:
                    st.error("No heat treatment parameters or mechanical properties found")

if __name__ == "__main__":
    main()
