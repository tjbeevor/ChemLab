import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="LPBF AlSi10Mg Analysis Dashboard", layout="wide")

def parse_lpbf_data(file):
    """Parse LPBF data with proper column handling"""
    try:
        # Read raw CSV first to examine structure
        df_raw = pd.read_csv(file)
        st.write("Raw columns:", df_raw.columns.tolist())
        
        # Read the CSV file with proper header handling
        df = pd.read_csv(file, skiprows=[0])  # Skip the merged cells row
        
        # Map columns based on actual data structure
        column_map = {
            'W': 'power',
            'mm/s': 'speed',
            'Hatch ': 'hatch',
            'thickness': 'thickness',
            'p/v': 'p_v_ratio',
            'Direction': 'build_direction',
            'solution temp': 'solution_temp',
            'ageing temp': 'ageing_temp',
            'Sol time': 'solution_time',
            'ageing time': 'ageing_time',
        }
        
        # Map UTS/YS columns
        uts_col = next((col for col in df.columns if 'UTS' in str(col)), None)
        if uts_col:
            column_map[uts_col] = 'uts'
        
        ys_col = next((col for col in df.columns if 'YS' in str(col)), None)
        if ys_col:
            column_map[ys_col] = 'ys'
            
        elongation_col = next((col for col in df.columns if 'Elongation' in str(col)), None)
        if elongation_col:
            column_map[elongation_col] = 'elongation'
            
        hardness_col = next((col for col in df.columns if 'hardness' in str(col) or 'HV' in str(col)), None)
        if hardness_col:
            column_map[hardness_col] = 'hardness'
        
        # Rename columns that exist in our mapping
        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        # Calculate energy density if possible
        if all(x in df.columns for x in ['power', 'speed', 'hatch', 'thickness']):
            mask = (df['speed'] > 0) & (df['hatch'] > 0) & (df['thickness'] > 0)
            df.loc[mask, 'energy_density'] = (
                df.loc[mask, 'power'] / 
                (df.loc[mask, 'speed'] * df.loc[mask, 'hatch'] * df.loc[mask, 'thickness'])
            )
        
        return df[list(column_map.values()) + ['energy_density']] if 'energy_density' in df.columns else df[list(column_map.values())]
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def main():
    st.title("🔬 LPBF AlSi10Mg Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes Laser Powder Bed Fusion (LPBF) process parameters 
    and their effects on AlSi10Mg mechanical properties.
    """)
    
    uploaded_file = st.file_uploader("Upload your LPBF data (CSV)", type='csv')
    
    if uploaded_file is not None:
        df = parse_lpbf_data(uploaded_file)
        
        if df is not None:
            st.write("Found data columns:", list(df.columns))
            
            # Analysis selection
            analysis = st.radio(
                "Select Analysis Type",
                ["Process-Property Relationships", "Heat Treatment Effects"]
            )
            
            if analysis == "Process-Property Relationships":
                process_params = ['power', 'speed', 'hatch', 'thickness', 'energy_density', 'p_v_ratio']
                properties = ['uts', 'ys', 'elongation', 'hardness']
                
                # Filter available parameters
                available_process = [p for p in process_params if p in df.columns]
                available_props = [p for p in properties if p in df.columns]
                
                if available_process and available_props:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_param = st.selectbox(
                            "Process Parameter",
                            options=available_process,
                            format_func=lambda x: x.replace('_', ' ').upper()
                        )
                    
                    with col2:
                        y_param = st.selectbox(
                            "Mechanical Property",
                            options=available_props,
                            format_func=lambda x: x.replace('_', ' ').upper()
                        )
                    
                    if x_param and y_param:
                        # Create basic scatter plot without trendline
                        fig = px.scatter(
                            df,
                            x=x_param,
                            y=y_param,
                            color='build_direction' if 'build_direction' in df.columns else None,
                            labels={
                                x_param: x_param.replace('_', ' ').upper(),
                                y_param: y_param.replace('_', ' ').upper()
                            },
                            title=f"{y_param.upper()} vs {x_param.upper()}"
                        )
                        
                        # Add custom layout
                        fig.update_layout(
                            height=600,
                            xaxis_title=f"{x_param.replace('_', ' ').upper()}",
                            yaxis_title=f"{y_param.replace('_', ' ').upper()}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate and show correlation
                        mask = df[[x_param, y_param]].notna().all(axis=1)
                        if mask.any():
                            corr = df[mask][x_param].corr(df[mask][y_param])
                            st.write(f"Correlation coefficient: {corr:.3f}")
                
            else:  # Heat Treatment Effects
                ht_params = ['solution_temp', 'ageing_temp', 'solution_time', 'ageing_time']
                mech_props = ['uts', 'ys', 'elongation', 'hardness']
                
                available_ht = [p for p in ht_params if p in df.columns]
                available_props = [p for p in mech_props if p in df.columns]
                
                if available_ht and available_props:
                    plot_cols = available_ht + available_props
                    plot_df = df[plot_cols].dropna()
                    
                    if not plot_df.empty:
                        fig = go.Figure(data=
                            go.Parcoords(
                                line=dict(
                                    color=plot_df[available_props[0]], 
                                    colorscale='Viridis'
                                ),
                                dimensions=[
                                    dict(
                                        range=[plot_df[col].min(), plot_df[col].max()],
                                        label=col.replace('_', ' ').upper(),
                                        values=plot_df[col]
                                    ) for col in plot_cols
                                ]
                            )
                        )
                        
                        fig.update_layout(
                            height=600,
                            title="Heat Treatment Parameters vs Mechanical Properties"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Insufficient data for heat treatment analysis")
                else:
                    st.error("No heat treatment parameters or mechanical properties found")

if __name__ == "__main__":
    main()
