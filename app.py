# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from pathlib import Path

# Configure the page
st.set_page_config(page_title="LPBF AlSi10Mg Analysis", layout="wide")

# Data Processing Functions
def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the CSV data"""
    df = pd.read_csv(uploaded_file)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Calculate energy density if not present
    if 'energy_density' not in df.columns:
        df['energy_density'] = df['power'] / (df['speed'] * df['hatch'] * df['thickness'])
    
    return df

def extract_features(df):
    """Extract relevant features for analysis"""
    process_params = ['power', 'speed', 'hatch', 'thickness', 'energy_density']
    ht_params = ['solution temp', 'sol time', 'ageing temp', 'ageing time']
    mech_props = ['uts', 'ys', 'elongation', 'hardness']
    
    return {
        'process': df[process_params],
        'heat_treatment': df[ht_params],
        'properties': df[mech_props]
    }

# Visualization Functions
def create_property_correlation_plot(df):
    """Create correlation matrix plot for mechanical properties"""
    corr = df.corr()
    fig = px.imshow(corr, 
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu")
    return fig

def create_process_property_plot(df, x_param, y_param, color_by='direction'):
    """Create scatter plot of process-property relationships"""
    fig = px.scatter(df, 
                    x=x_param,
                    y=y_param,
                    color=color_by,
                    title=f"{y_param.upper()} vs {x_param.upper()}",
                    labels={x_param: x_param.upper(), 
                           y_param: y_param.upper()})
    return fig

def create_parallel_coordinates_plot(df):
    """Create parallel coordinates plot for heat treatment analysis"""
    dimensions = ['solution temp', 'sol time', 'ageing temp', 
                 'ageing time', 'uts', 'ys', 'elongation']
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=df['uts'],
                     colorscale='Viridis'),
            dimensions=[dict(range=[df[dim].min(), df[dim].max()],
                           label=dim.upper(),
                           values=df[dim]) for dim in dimensions]
        )
    )
    return fig

# Machine Learning Functions
def train_ml_model(df, target='uts'):
    """Train Random Forest model for property prediction"""
    features = ['power', 'speed', 'hatch', 'thickness', 
                'solution temp', 'sol time', 'ageing temp', 'ageing time']
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, (r2, rmse), (X_test, y_test, y_pred)

def plot_feature_importance(model, feature_names):
    """Plot feature importance from trained model"""
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(importances, x='importance', y='feature', 
                 orientation='h', title='Feature Importance')
    return fig

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
        st.session_state.data = load_and_preprocess_data(uploaded_file)
    
    if st.session_state.data is not None:
        if page == "Data Upload & Overview":
            st.subheader("Data Overview")
            st.write(st.session_state.data.head())
            st.write("Dataset Shape:", st.session_state.data.shape)
            
            # Basic statistics
            st.subheader("Statistical Summary")
            st.write(st.session_state.data.describe())
            
        elif page == "Process-Property Analysis":
            st.subheader("Process-Property Relationships")
            
            col1, col2 = st.columns(2)
            with col1:
                x_param = st.selectbox("X-axis Parameter",
                    ['power', 'speed', 'energy_density', 'hatch', 'thickness'])
            with col2:
                y_param = st.selectbox("Y-axis Parameter",
                    ['uts', 'ys', 'elongation', 'hardness'])
            
            fig = create_process_property_plot(
                st.session_state.data, x_param, y_param)
            st.plotly_chart(fig, use_container_width=True)
            
        elif page == "Heat Treatment Analysis":
            st.subheader("Heat Treatment Analysis")
            
            fig = create_parallel_coordinates_plot(st.session_state.data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif page == "Statistical Analysis":
            st.subheader("Statistical Analysis")
            
            # Correlation matrix
            st.write("Correlation Matrix")
            fig = create_property_correlation_plot(st.session_state.data)
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA Analysis
            st.write("Principal Component Analysis")
            features = extract_features(st.session_state.data)
            pca = PCA()
            pca_result = pca.fit_transform(StandardScaler().fit_transform(
                features['process']))
            
            explained_variance = pd.DataFrame(
                pca.explained_variance_ratio_,
                columns=['Explained Variance Ratio']
            )
            st.write(explained_variance)
            
        elif page == "Machine Learning":
            st.subheader("Property Prediction Model")
            
            target_prop = st.selectbox(
                "Select Property to Predict",
                ['uts', 'ys', 'elongation', 'hardness']
            )
            
            if st.button("Train Model"):
                model, metrics, predictions = train_ml_model(
                    st.session_state.data, target_prop)
                
                st.write(f"Model R² Score: {metrics[0]:.3f}")
                st.write(f"Model RMSE: {metrics[1]:.3f}")
                
                # Feature importance plot
                fig = plot_feature_importance(
                    model,
                    ['power', 'speed', 'hatch', 'thickness', 
                     'solution temp', 'sol time', 'ageing temp', 'ageing time']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Predictions vs Actual plot
                X_test, y_test, y_pred = predictions
                fig = px.scatter(x=y_test, y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title=f'Predicted vs Actual {target_prop.upper()}')
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction'
                ))
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
