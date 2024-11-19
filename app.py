import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="LPBF AlSi10Mg Analysis Dashboard", layout="wide")

def parse_lpbf_data(file):
    """Parse LPBF data with proper column handling"""
    # Read the CSV file
    df = pd.read_csv(file, header=[0,1])
    
    # Extract the relevant columns and rename them appropriately
    column_mapping = {
        # Process parameters
        ('LPBF process  parameter', 'power'): 'power',
        ('LPBF process  parameter', 'speed'): 'speed',
        ('LPBF process  parameter', 'Hatch '): 'hatch',
        ('LPBF process  parameter', 'thickness'): 'thickness',
        ('LPBF process  parameter', 'p/v'): 'p_v_ratio',
        ('LPBF process  parameter', 'Direction'): 'build_direction',
        
        # Heat treatment parameters
        ('Heat Treatment Method', 'solution temp'): 'solution_temp',
        ('Heat Treatment Method', 'ageing temp'): 'ageing_temp',
        ('Heat Treatment Method', 'Sol time'): 'solution_time',
        ('Heat Treatment Method', 'ageing time'): 'ageing_time',
        
        # Mechanical properties
        ('Heat Treated Properties', 'UTS'): 'uts_ht',
        ('Heat Treated Properties', 'YS'): 'ys_ht',
        ('Heat Treated Properties', 'Elongation'): 'elongation_ht',
        ('Heat Treated Properties', 'hardness'): 'hardness_ht',
        ('As Built Properties', 'UTS'): 'uts_ab',
        ('As Built Properties', 'YS'): 'ys_ab',
        ('As Built Properties', 'Elongation'): 'elongation_ab',
        ('As Built Properties', 'hardness'): 'hardness_ab'
    }
    
    # Create new dataframe with mapped columns
    processed_df = pd.DataFrame()
    
    for (old_header, old_subheader), new_name in column_mapping.items():
        try:
            processed_df[new_name] = pd.to_numeric(df[old_header][old_subheader], errors='coerce')
        except:
            continue
    
    # Calculate energy density
    if all(x in processed_df.columns for x in ['power', 'speed', 'hatch', 'thickness']):
        mask = (processed_df['speed'] > 0) & (processed_df['hatch'] > 0) & (processed_df['thickness'] > 0)
        processed_df.loc[mask, 'energy_density'] = (
            processed_df.loc[mask, 'power'] / 
            (processed_df.loc[mask, 'speed'] * processed_df.loc[mask, 'hatch'] * processed_df.loc[mask, 'thickness'])
        )
    
    return processed_df

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
                properties = [
                    'uts_ab', 'ys_ab', 'elongation_ab', 'hardness_ab',
                    'uts_ht', 'ys_ht', 'elongation_ht', 'hardness_ht'
                ]
                
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
                        fig = px.scatter(
                            df,
                            x=x_param,
                            y=y_param,
                            color='build_direction' if 'build_direction' in df.columns else None,
                            trendline="ols",
                            labels={
                                x_param: x_param.replace('_', ' ').upper(),
                                y_param: y_param.replace('_', ' ').upper()
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show correlation
                        corr = df[x_param].corr(df[y_param])
                        st.write(f"Correlation coefficient: {corr:.3f}")
                
            else:  # Heat Treatment Effects
                ht_params = ['solution_temp', 'ageing_temp', 'solution_time', 'ageing_time']
                mech_props = ['uts_ht', 'ys_ht', 'elongation_ht', 'hardness_ht']
                
                available_ht = [p for p in ht_params if p in df.columns]
                available_props = [p for p in mech_props if p in df.columns]
                
                if available_ht and available_props:
                    plot_cols = available_ht + available_props
                    plot_df = df[plot_cols].dropna()
                    
                    if not plot_df.empty:
                        fig = go.Figure(data=
                            go.Parcoords(
                                line=dict(color=plot_df[available_props[0]], 
                                         colorscale='Viridis'),
                                dimensions=[
                                    dict(
                                        range=[plot_df[col].min(), plot_df[col].max()],
                                        label=col.replace('_', ' ').upper(),
                                        values=plot_df[col]
                                    ) for col in plot_cols
                                ]
                            )
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Insufficient data for heat treatment analysis")
                else:
                    st.error("No heat treatment parameters or mechanical properties found")

if __name__ == "__main__":
    main()
```
