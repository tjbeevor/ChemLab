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

def create_property_correlation_plot(df):
    """Create correlation matrix plot for mechanical properties"""
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create heatmap using plotly
    fig = px.imshow(
        corr,
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_tickangle=-45,
        width=800,
        height=800
    )
    
    return fig

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

def train_ml_model(df, target='uts'):
    """Train Random Forest model for property prediction"""
    features = ['power', 'speed', 'hatch', 'thickness', 
                'solution temp', 'sol time', 'ageing temp', 'ageing time']
    
    # Check if required columns exist
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        raise ValueError("No required feature columns found in the dataset")
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the dataset")
    
    # Use only available features
    X = df[available_features]
    y = df[target].dropna()
    
    # Check if we have enough data
    if len(y) < 10:
        raise ValueError("Insufficient data points for training (minimum 10 required)")
    
    # Remove rows with NaN values
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, (r2, rmse), (X_test, y_test, y_pred), available_features

def plot_feature_importance(model, feature_names):
    """Plot feature importance from trained model"""
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(importances, 
                 x='importance', 
                 y='feature', 
                 orientation='h', 
                 title='Feature Importance')
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
        try:
            st.session_state.data = load_and_preprocess_data(uploaded_file)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.data = None
    
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
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
        elif page == "Heat Treatment Analysis":
            st.subheader("Heat Treatment Analysis")
            
            fig = create_parallel_coordinates_plot(st.session_state.data)
            if fig:
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
            if not features['process'].empty:
                pca = PCA()
                pca_result = pca.fit_transform(StandardScaler().fit_transform(
                    features['process']))
                
                explained_variance = pd.DataFrame(
                    pca.explained_variance_ratio_,
                    columns=['Explained Variance Ratio']
                )
                st.write(explained_variance)
            else:
                st.warning("Insufficient process parameters for PCA analysis")
            
        elif page == "Machine Learning":
            st.subheader("Property Prediction Model")
            
            # Only show properties that exist in the data
            available_properties = [prop for prop in ['uts', 'ys', 'elongation', 'hardness'] 
                                 if prop in st.session_state.data.columns]
            
            if not available_properties:
                st.error("No mechanical properties found in the dataset")
            else:
                target_prop = st.selectbox(
                    "Select Property to Predict",
                    available_properties
                )
                
                if st.button("Train Model"):
                    try:
                        model, metrics, predictions, used_features = train_ml_model(
                            st.session_state.data, target_prop)
                        
                        st.write(f"Model R² Score: {metrics[0]:.3f}")
                        st.write(f"Model RMSE: {metrics[1]:.3f}")
                        
                        # Feature importance plot
                        fig = plot_feature_importance(model, used_features)
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
                    
                    except ValueError as e:
                        st.error(f"Error training model: {str(e)}")
                    except Exception as e:
                        st.error(f"Unexpected error during model training: {str(e)}")

if __name__ == "__main__":
    main()
