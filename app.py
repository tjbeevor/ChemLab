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

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the CSV data"""
    # Read the first few rows to understand the structure
    df = pd.read_csv(uploaded_file, header=[0,1])  # Read with multi-level headers
    
    # Flatten column names
    df.columns = [f"{'' if pd.isna(c[0]) else c[0]}_{'' if pd.isna(c[1]) else c[1]}".strip('_').lower() for c in df.columns]
    
    # Clean column names
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('(', '').str.replace(')', '')
    df.columns = df.columns.str.replace('%', 'percent')
    
    # Show column names for debugging
    st.write("Cleaned columns:", sorted(list(df.columns)))
    
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^unnamed')]
    
    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            continue
    
    # Handle direction column if it exists
    direction_cols = [col for col in df.columns if 'direction' in col.lower()]
    if direction_cols:
        df[direction_cols[0]] = df[direction_cols[0]].str.upper() if df[direction_cols[0]].dtype == object else df[direction_cols[0]]
    
    # Handle numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Calculate energy density if possible
    power_cols = [col for col in df.columns if 'power' in col.lower() or 'w' in col.lower()]
    speed_cols = [col for col in df.columns if 'speed' in col.lower() or 'mm/s' in col.lower()]
    hatch_cols = [col for col in df.columns if 'hatch' in col.lower()]
    thickness_cols = [col for col in df.columns if 'thickness' in col.lower()]
    
    if power_cols and speed_cols and hatch_cols and thickness_cols:
        try:
            df['energy_density'] = df[power_cols[0]] / (df[speed_cols[0]] * df[hatch_cols[0]] * df[thickness_cols[0]])
        except:
            df['energy_density'] = np.nan
    
    return df
def create_property_correlation_plot(df):
    """Create correlation matrix plot"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr,
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(
        width=800,
        height=800
    )
    
    return fig

def create_process_property_plot(df, x_param, y_param, color_by='direction'):
    """Create scatter plot"""
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
    # Find relevant columns
    ht_cols = [col for col in df.columns if any(x in col for x in ['temp', 'time', 'uts', 'ys', 'elongation'])]
    
    if not ht_cols:
        return None
        
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=df[ht_cols[0]], colorscale='Viridis'),
            dimensions=[dict(range=[df[dim].min(), df[dim].max()],
                           label=dim.upper(),
                           values=df[dim]) for dim in ht_cols]
        )
    )
    
    fig.update_layout(height=600)
    return fig

def train_ml_model(df, target='uts'):
    """Train Random Forest model"""
    # Select numeric columns as features
    feature_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in feature_cols if col != target]
    
    if not feature_cols:
        raise ValueError("No numeric feature columns found")
    
    X = df[feature_cols]
    y = df[target]
    
    # Remove rows with NaN values
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask]
    y = y[mask]
    
    if len(y) < 10:
        raise ValueError("Insufficient data points for training")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, (r2, rmse), (X_test, y_test, y_pred), feature_cols

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(importances, x='importance', y='feature', 
                 orientation='h', title='Feature Importance')
    return fig

def main():
    st.title("LPBF AlSi10Mg Analysis Dashboard")
    
    # Sidebar navigation
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
        df = st.session_state.data
        
        if page == "Data Upload & Overview":
            st.subheader("Data Overview")
            st.write("Available columns:", sorted(list(df.columns)))
            st.write("Dataset Shape:", df.shape)
            st.write(df.head())
            
            st.subheader("Statistical Summary")
            st.write(df.describe())
            
        elif page == "Process-Property Analysis":
            st.subheader("Process-Property Relationships")
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            col1, col2 = st.columns(2)
            with col1:
                x_param = st.selectbox("X-axis Parameter", numeric_cols)
            with col2:
                y_param = st.selectbox("Y-axis Parameter", numeric_cols)
            
            fig = create_process_property_plot(df, x_param, y_param)
            st.plotly_chart(fig, use_container_width=True)
            
        elif page == "Heat Treatment Analysis":
            st.subheader("Heat Treatment Analysis")
            
            fig = create_parallel_coordinates_plot(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for heat treatment analysis")
            
        elif page == "Statistical Analysis":
            st.subheader("Statistical Analysis")
            
            st.write("Correlation Matrix")
            fig = create_property_correlation_plot(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Principal Component Analysis")
            numeric_df = df.select_dtypes(include=[np.number])
            
            if not numeric_df.empty:
                # Remove constant columns and handle NaN values
                numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
                numeric_df = numeric_df.fillna(numeric_df.mean())
                
                if numeric_df.shape[1] > 1:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_df)
                    
                    pca = PCA()
                    pca_result = pca.fit_transform(scaled_data)
                    
                    explained_variance = pd.DataFrame(
                        pca.explained_variance_ratio_,
                        columns=['Explained Variance Ratio']
                    )
                    st.write(explained_variance)
                else:
                    st.warning("Insufficient numeric variables for PCA")
            else:
                st.warning("No numeric columns found for PCA")
            
        elif page == "Machine Learning":
            st.subheader("Property Prediction Model")
            
            # Get numeric columns as potential targets
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                target_prop = st.selectbox(
                    "Select Property to Predict",
                    numeric_cols
                )
                
                if st.button("Train Model"):
                    try:
                        model, metrics, predictions, features = train_ml_model(
                            df, target_prop)
                        
                        st.write(f"Model R² Score: {metrics[0]:.3f}")
                        st.write(f"Model RMSE: {metrics[1]:.3f}")
                        
                        fig = plot_feature_importance(model, features)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        X_test, y_test, y_pred = predictions
                        fig = px.scatter(
                            x=y_test, y=y_pred,
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title=f'Predicted vs Actual {target_prop.upper()}'
                        )
                        fig.add_trace(go.Scatter(
                            x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode='lines',
                            name='Perfect Prediction'
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error in model training: {str(e)}")
            else:
                st.error("No numeric columns found for modeling")

if __name__ == "__main__":
    main()
