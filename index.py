# Malaria Risk Predictor Application with Interactive Map

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import os
import hashlib
from datetime import datetime
import pickle
import folium
from streamlit_folium import st_folium
import random

# Page configuration
st.set_page_config(
    page_title="Malaria Risk Predictor",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    h1, h2, h3 {
        color: white;
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #E55A2B;
    }
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .risk-high {
        color: #FF6B35;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-medium {
        color: #FFA500;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-low {
        color: #4CAF50;
        font-size: 1.5rem;
        font-weight: bold;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    </style>
""", unsafe_allow_html=True)

# File paths
USERS_FILE = "users.json"
MODEL_FILE = "malaria_model.pkl"
HISTORY_FILE = "prediction_history.json"

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "malaria_model" not in st.session_state:
    st.session_state.malaria_model = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Map Predictor"
if "selected_location" not in st.session_state:
    st.session_state.selected_location = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# User management functions
def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {
        "password": hash_password(password),
        "created_at": datetime.now().isoformat()
    }
    save_users(users)
    return True, "Registration successful!"

def authenticate_user(username, password):
    """Authenticate a user"""
    users = load_users()
    if username not in users:
        return False, "Username not found"
    if users[username]["password"] != hash_password(password):
        return False, "Incorrect password"
    return True, "Login successful!"

# Prediction history functions
def load_history():
    """Load prediction history"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_history(history):
    """Save prediction history"""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def add_prediction_to_history(username, location, features, prediction, risk_level, confidence):
    """Add a prediction to history"""
    history = load_history()
    if username not in history:
        history[username] = []
    
    history[username].append({
        "timestamp": datetime.now().isoformat(),
        "location": location,
        "features": features,
        "prediction": prediction,
        "risk_level": risk_level,
        "confidence": confidence
    })
    
    save_history(history)

def get_user_history(username):
    """Get prediction history for a user"""
    history = load_history()
    return history.get(username, [])

# Feature extraction from coordinates
def extract_features_from_location(lat, lon):
    """Extract environmental features from geographic coordinates"""
    # Simulate feature extraction from APIs/datasets
    # In a real application, this would call actual APIs for:
    # - Rainfall data (e.g., from weather APIs)
    # - Temperature data
    # - NDVI (vegetation index)
    # - Population density
    # - Elevation
    # - Water body coverage
    
    # For demonstration, we'll generate realistic synthetic data
    np.random.seed(int(lat * 1000 + lon * 1000))
    
    # Base values that vary by location
    base_rainfall = 800 + (lat - 10) * 50 + np.random.normal(0, 200)
    base_temp = 25 + (lat - 10) * 0.5 + np.random.normal(0, 3)
    base_ndvi = 0.3 + (lat - 10) * 0.01 + np.random.normal(0, 0.1)
    base_pop = max(0, 50 + (lat - 10) * 10 + np.random.normal(0, 30))
    base_elevation = max(0, 500 + (lat - 10) * 50 + np.random.normal(0, 200))
    base_water = max(0, min(100, 20 + (lat - 10) * 2 + np.random.normal(0, 15)))
    
    features = {
        "rainfall_12mo": round(max(0, base_rainfall), 2),
        "temp_mean": round(base_temp, 2),
        "env_temp_mean": round(base_temp, 2),  # Alias for compatibility
        "ndvi_mean": round(max(0, min(1, base_ndvi)), 2),
        "pop_density": round(base_pop, 2),
        "elevation": round(base_elevation, 2),
        "water_coverage": round(base_water, 2)
    }
    
    return features

# Malaria prediction model - supports both patient and environmental features
def create_malaria_model():
    """Create and train a malaria risk prediction model using both patient and environmental features"""
    np.random.seed(42)
    n_samples = 2000
    
    # Combined features: patient data + environmental data
    # Patient: age, body_temp, recent_travel, mosquito_exposure, symptoms, preventive_medication, region_risk, immune_status
    # Environmental: rainfall_12mo, env_temp_mean, ndvi_mean, pop_density, elevation, water_coverage
    data = []
    for _ in range(n_samples):
        # Patient features
        age = np.random.randint(1, 80)
        body_temp = np.random.normal(37.0, 1.0)
        recent_travel = np.random.choice([0, 1], p=[0.7, 0.3])
        mosquito_exposure = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        symptoms = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        preventive_medication = np.random.choice([0, 1], p=[0.6, 0.4])
        region_risk = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        immune_status = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Environmental features
        rainfall = np.random.uniform(200, 2000)
        env_temp_mean = np.random.uniform(15, 35)
        ndvi_mean = np.random.uniform(0, 1)
        pop_density = np.random.uniform(0, 500)
        elevation = np.random.uniform(0, 3000)
        water_coverage = np.random.uniform(0, 100)
        
        # Calculate risk score
        patient_risk = (
            (body_temp - 37.0) * 2 +
            recent_travel * 3 +
            mosquito_exposure * 2 +
            symptoms * 2 +
            (1 - preventive_medication) * 2 +
            region_risk * 2 +
            (1 - immune_status) * 1.5
        )
        
        env_temp_factor = 1.0 if 20 <= env_temp_mean <= 30 else 0.5
        rainfall_factor = min(1.0, rainfall / 1500)
        ndvi_factor = ndvi_mean
        pop_factor = min(1.0, pop_density / 200)
        elevation_factor = 1.0 if elevation < 1000 else max(0.3, 1.0 - (elevation - 1000) / 2000)
        water_factor = min(1.0, water_coverage / 50)
        
        env_risk = (
            env_temp_factor * 2 +
            rainfall_factor * 2 +
            ndvi_factor * 1.5 +
            pop_factor * 1.5 +
            elevation_factor * 1.5 +
            water_factor * 1.5
        ) * 2
        
        total_risk = (patient_risk + env_risk) / 20
        
        # Convert to binary label
        label = 1 if total_risk > 0.4 else 0
        
        data.append([
            age, body_temp, recent_travel, mosquito_exposure, symptoms,
            preventive_medication, region_risk, immune_status,
            rainfall, env_temp_mean, ndvi_mean, pop_density, elevation, water_coverage, label
        ])
    
    df = pd.DataFrame(data, columns=[
        "age", "body_temp", "recent_travel", "mosquito_exposure", "symptoms",
        "preventive_medication", "region_risk", "immune_status",
        "rainfall_12mo", "env_temp_mean", "ndvi_mean", "pop_density", 
        "elevation", "water_coverage", "risk"
    ])
    
    X = df.drop("risk", axis=1)
    y = df["risk"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def load_or_create_model():
    """Load existing model or create a new one"""
    if st.session_state.malaria_model is not None:
        return st.session_state.malaria_model
    
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
    else:
        model = create_malaria_model()
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
    
    st.session_state.malaria_model = model
    return model

def predict_malaria_risk_from_patient_data(age, body_temp, recent_travel, mosquito_exposure, 
                                          symptoms, preventive_medication, region_risk, immune_status,
                                          env_features=None):
    """Predict malaria risk based on patient data and optional environmental features"""
    model = load_or_create_model()
    
    # If no environmental features provided, use defaults
    if env_features is None:
        env_features = {
            "rainfall_12mo": 1000.0,
            "env_temp_mean": 25.0,
            "ndvi_mean": 0.5,
            "pop_density": 100.0,
            "elevation": 500.0,
            "water_coverage": 20.0
        }
    
    feature_array = np.array([[
        age, body_temp, recent_travel, mosquito_exposure, symptoms,
        preventive_medication, region_risk, immune_status,
        env_features["rainfall_12mo"],
        env_features["env_temp_mean"],
        env_features["ndvi_mean"],
        env_features["pop_density"],
        env_features["elevation"],
        env_features["water_coverage"]
    ]])
    
    risk_probability = model.predict_proba(feature_array)[0][1]
    risk_class = model.predict(feature_array)[0]
    
    return risk_class, risk_probability

def predict_malaria_risk_from_features(features):
    """Predict malaria risk based on environmental features (with default patient data)"""
    model = load_or_create_model()
    
    # Use default patient values when only environmental features are provided
    default_patient = {
        "age": 30,
        "body_temp": 37.0,
        "recent_travel": 0,
        "mosquito_exposure": 1,
        "symptoms": 0,
        "preventive_medication": 0,
        "region_risk": 1,
        "immune_status": 0
    }
    
    feature_array = np.array([[
        default_patient["age"], default_patient["body_temp"], 
        default_patient["recent_travel"], default_patient["mosquito_exposure"],
        default_patient["symptoms"], default_patient["preventive_medication"],
        default_patient["region_risk"], default_patient["immune_status"],
        features["rainfall_12mo"],
        features["temp_mean"],
        features["ndvi_mean"],
        features["pop_density"],
        features["elevation"],
        features["water_coverage"]
    ]])
    
    risk_probability = model.predict_proba(feature_array)[0][1]
    risk_class = model.predict(feature_array)[0]
    
    return risk_class, risk_probability

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability >= 0.7:
        return "High", "🔴"
    elif probability >= 0.4:
        return "Medium", "🟡"
    else:
        return "Low", "🟢"

# Login page
def login_page():
    """Display login page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center; color: white;'>🦟 Malaria Risk Predictor</h1>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.markdown("<h2 style='color: white;'>Login</h2>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_btn"):
                if username and password:
                    success, message = authenticate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.current_page = "Personal Entry"
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter both username and password")
        
        with tab2:
            st.markdown("<h2 style='color: white;'>Register</h2>", unsafe_allow_html=True)
            new_username = st.text_input("Username", key="reg_username")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            if st.button("Register", key="register_btn"):
                if new_username and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        success, message = register_user(new_username, new_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill in all fields")

# Sidebar navigation
def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.markdown(f"## Welcome, {st.session_state.username}! 👋")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Navigate to:")
    page = st.sidebar.radio(
        "Select page",
        ["Personal Entry", "Map Predictor", "Prediction History", "Account Info"],
        key="page_selector"
    )
    
    st.session_state.current_page = page
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", key="sidebar_logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.current_page = "Personal Entry"
        st.session_state.selected_location = None
        st.session_state.prediction_result = None
        st.rerun()

# Personal Entry page
def personal_entry_page():
    """Display personal entry form for patient information"""
    st.markdown("### 📝 Enter Patient Information")
    st.markdown("Fill in the patient details below to get a personalized malaria risk assessment.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h4 style='color: white;'>Patient Information</h4>", unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=100, value=30, key="patient_age")
        body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1, key="patient_temp")
        recent_travel = st.selectbox("Recent Travel to Malaria-Endemic Area", 
                                     ["No", "Yes"], key="patient_travel")
        mosquito_exposure = st.selectbox("Mosquito Exposure Level", 
                                        ["Low", "Medium", "High"], key="patient_mosquito")
        symptoms = st.number_input("Number of Symptoms (fever, chills, headache, etc.)", 
                                  min_value=0, max_value=10, value=0, key="patient_symptoms")
    
    with col2:
        st.markdown("<h4 style='color: white;'>Additional Factors</h4>", unsafe_allow_html=True)
        preventive_medication = st.selectbox("Taking Preventive Medication", 
                                           ["No", "Yes"], key="patient_medication")
        region_risk = st.selectbox("Current Region Risk Level", 
                                  ["Low", "Medium", "High"], key="patient_region")
        immune_status = st.selectbox("Immune Status", 
                                   ["No Previous Exposure", "Some Immunity"], key="patient_immune")
        
        # Optional: Link to map for environmental data
        st.markdown("---")
        st.markdown("<small style='color: white;'>💡 Tip: You can also use the Map Predictor to get location-based environmental data</small>", unsafe_allow_html=True)
    
    if st.button("Predict Malaria Risk", key="patient_predict_btn"):
        # Convert inputs to model format
        recent_travel_val = 1 if recent_travel == "Yes" else 0
        mosquito_exposure_map = {"Low": 0, "Medium": 1, "High": 2}
        mosquito_exposure_val = mosquito_exposure_map[mosquito_exposure]
        preventive_medication_val = 1 if preventive_medication == "Yes" else 0
        region_risk_map = {"Low": 0, "Medium": 1, "High": 2}
        region_risk_val = region_risk_map[region_risk]
        immune_status_val = 1 if immune_status == "Some Immunity" else 0
        
        # Check if we have environmental features from map selection
        env_features = None
        if st.session_state.get("extracted_features"):
            env_features = {
                "rainfall_12mo": st.session_state.extracted_features.get("rainfall_12mo", 1000.0),
                "env_temp_mean": st.session_state.extracted_features.get("temp_mean", 25.0),
                "ndvi_mean": st.session_state.extracted_features.get("ndvi_mean", 0.5),
                "pop_density": st.session_state.extracted_features.get("pop_density", 100.0),
                "elevation": st.session_state.extracted_features.get("elevation", 500.0),
                "water_coverage": st.session_state.extracted_features.get("water_coverage", 20.0)
            }
            st.info("ℹ️ Using environmental data from your last map selection")
        
        try:
            risk_class, risk_probability = predict_malaria_risk_from_patient_data(
                age, body_temp, recent_travel_val, mosquito_exposure_val,
                symptoms, preventive_medication_val, region_risk_val, immune_status_val,
                env_features
            )
            
            risk_level, risk_emoji = get_risk_level(risk_probability)
            
            st.markdown("---")
            st.markdown("### 🦟 Prediction Result")
            
            col_pred1, col_pred2 = st.columns([2, 1])
            
            with col_pred1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 class="risk-{risk_level.lower()}">{risk_emoji} Malaria Risk: {risk_level}</h3>
                    <p style='color: white; font-size: 1.2rem;'>Risk Probability: <b>{(risk_probability * 100):.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_pred2:
                if risk_level == "High":
                    st.error("⚠️ High Risk")
                elif risk_level == "Medium":
                    st.warning("⚠️ Medium Risk")
                else:
                    st.success("✅ Low Risk")
            
            # Recommendations
            st.markdown("### Recommendations")
            if risk_level == "High":
                st.warning("⚠️ **Immediate Action Recommended:** Please consult a healthcare professional immediately. Consider getting a malaria test.")
            elif risk_level == "Medium":
                st.info("ℹ️ **Precautionary Measures:** Monitor symptoms closely. Consider preventive measures and consult a doctor if symptoms worsen.")
            else:
                st.success("✅ **Low Risk:** Continue preventive measures. Stay vigilant about mosquito protection.")
            
            # Save to history
            prediction_data = {
                "age": age,
                "body_temp": body_temp,
                "recent_travel": recent_travel,
                "mosquito_exposure": mosquito_exposure,
                "symptoms": symptoms,
                "preventive_medication": preventive_medication,
                "region_risk": region_risk,
                "immune_status": immune_status
            }
            if env_features:
                prediction_data.update(env_features)
            
            add_prediction_to_history(
                st.session_state.username,
                "Personal Entry",
                prediction_data,
                risk_level,
                risk_level,
                risk_probability
            )
            
            # Additional information
            with st.expander("View Detailed Analysis"):
                st.write(f"**Age:** {age} years")
                st.write(f"**Body Temperature:** {body_temp}°C")
                st.write(f"**Recent Travel:** {recent_travel}")
                st.write(f"**Mosquito Exposure:** {mosquito_exposure}")
                st.write(f"**Symptoms Count:** {symptoms}")
                st.write(f"**Preventive Medication:** {preventive_medication}")
                st.write(f"**Region Risk:** {region_risk}")
                st.write(f"**Immune Status:** {immune_status}")
                if env_features:
                    st.write("**Environmental Factors (from map):**")
                    st.write(f"- Rainfall: {env_features['rainfall_12mo']} mm")
                    st.write(f"- Temperature: {env_features['env_temp_mean']}°C")
                    st.write(f"- NDVI: {env_features['ndvi_mean']:.2f}")
                    st.write(f"- Population Density: {env_features['pop_density']:.1f}/km²")
                    st.write(f"- Elevation: {env_features['elevation']:.1f} m")
                    st.write(f"- Water Coverage: {env_features['water_coverage']:.1f}%")
                st.write(f"**Risk Score:** {risk_probability:.3f}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Map Predictor page
def map_predictor_page():
    """Display the interactive map predictor page"""
    st.markdown("### Click on the map to select a location")
    
    # Create map centered on Africa
    m = folium.Map(
        location=[8, 20],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Add click handler
    folium.LatLngPopup().add_to(m)
    
    # Display map and get click data
    map_data = st_folium(m, width=1200, height=600, key="africa_map")
    
    # Check if map was clicked
    if map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        
        st.session_state.selected_location = {"lat": lat, "lon": lon}
        
        # Extract features
        with st.spinner("Extracting features..."):
            features = extract_features_from_location(lat, lon)
            st.session_state.extracted_features = features
        
        # Display extracted features
        st.markdown("### Extracted Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🌧️ Rainfall (12mo)", f"{features['rainfall_12mo']} mm")
            st.metric("🌡️ Temperature", f"{features['temp_mean']}°C")
        
        with col2:
            st.metric("🌿 NDVI Mean", f"{features['ndvi_mean']:.2f}")
            st.metric("👥 Population Density", f"{features['pop_density']:.1f}/km²")
        
        with col3:
            st.metric("⛰️ Elevation", f"{features['elevation']:.1f} m")
            st.metric("💧 Water Coverage", f"{features['water_coverage']:.1f}%")
        
        # Make prediction
        try:
            risk_class, risk_probability = predict_malaria_risk_from_features(features)
            risk_level, risk_emoji = get_risk_level(risk_probability)
            
            st.session_state.prediction_result = {
                "location": {"lat": lat, "lon": lon},
                "features": features,
                "risk_level": risk_level,
                "confidence": risk_probability
            }
            
            # Display prediction result
            st.markdown("---")
            st.markdown("### 🦟 Prediction Result")
            
            col_pred1, col_pred2 = st.columns([2, 1])
            
            with col_pred1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 class="risk-{risk_level.lower()}">{risk_emoji} Malaria Risk: {risk_level}</h3>
                    <p style='color: white; font-size: 1.2rem;'>Confidence: <b>{(risk_probability * 100):.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_pred2:
                # Confidence gauge
                confidence_pct = risk_probability * 100
                if risk_level == "High":
                    st.error(f"⚠️ High Risk Area")
                elif risk_level == "Medium":
                    st.warning(f"⚠️ Medium Risk Area")
                else:
                    st.success(f"✅ Low Risk Area")
            
            # Generate historical trend data (last 5 years)
            np.random.seed(int(lat * 1000 + lon * 1000))
            years = [2020, 2021, 2022, 2023, 2024]
            
            # Generate rainfall trend (vary around the current value)
            base_rainfall = features['rainfall_12mo']
            rainfall_trend = [max(0, base_rainfall + np.random.normal(0, 100)) for _ in years]
            
            # Generate temperature trend (vary around the current value)
            base_temp = features['temp_mean']
            temp_trend = [max(0, base_temp + np.random.normal(0, 2)) for _ in years]
            
            # Display annual trends charts
            st.markdown("---")
            st.markdown("### 📈 Environmental Trends (Last 5 Years)")
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                rainfall_df = pd.DataFrame({
                    'Year': years,
                    'Rainfall (mm)': rainfall_trend
                })
                st.line_chart(rainfall_df.set_index('Year'), y_label="Rainfall (mm)", height=300)
                st.caption("Annual Rainfall Trend (Last 5 Years)")
            
            with trend_col2:
                temp_df = pd.DataFrame({
                    'Year': years,
                    'Temperature (°C)': temp_trend
                })
                st.line_chart(temp_df.set_index('Year'), y_label="Temperature (°C)", height=300)
                st.caption("Annual Temperature Trend (Last 5 Years)")
            
            # Display environmental features bar chart
            st.markdown("---")
            st.markdown("### 📊 Environmental Features at Selected Location")
            
            # Prepare data for bar chart
            feature_data = {
                'Feature': [
                    'Rainfall (mm)',
                    'Temperature (°C)',
                    'Vegetation (NDVI)',
                    'Population Density',
                    'Elevation (m)',
                    'Water Coverage (%)'
                ],
                'Value': [
                    features['rainfall_12mo'],
                    features['temp_mean'],
                    features['ndvi_mean'] * 100,  # Scale NDVI for visibility
                    features['pop_density'],
                    features['elevation'],
                    features['water_coverage']
                ]
            }
            
            feature_df = pd.DataFrame(feature_data)
            st.bar_chart(feature_df.set_index('Feature'), height=400)
            
            # Risk Interpretation section
            st.markdown("---")
            st.markdown("### 🔍 Risk Interpretation")
            
            if risk_level == "High":
                st.error("""
                **High Risk Area - Immediate Action Required**
                - Immediate medical consultation recommended
                - Consider preventive medication
                - Enhanced mosquito protection measures
                - Regular health monitoring
                - Community-wide intervention programs
                """)
            elif risk_level == "Medium":
                st.warning("""
                **Medium Risk Area - Consider**
                - Seasonal monitoring
                - Basic preventive measures
                - Community awareness programs
                - Regular health check-ups
                - Mosquito control initiatives
                """)
            else:
                st.success("""
                **Low Risk Area - Maintain Vigilance**
                - Continue standard preventive measures
                - Monitor environmental changes
                - Stay informed about local health advisories
                - Maintain good hygiene practices
                - Support community health programs
                """)
            
            # Save to history
            add_prediction_to_history(
                st.session_state.username,
                {"lat": lat, "lon": lon},
                features,
                risk_level,
                risk_level,
                risk_probability
            )
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # How to use section
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Zoom to your area of interest**
        2. **Click on any location on the map**
        3. **Wait for feature extraction**
        4. **View prediction results and charts**
        
        **The system will analyze:**
        - 🌧️ Rainfall patterns
        - 🌡️ Temperature data
        - 🌿 Vegetation (NDVI)
        - 👥 Population density
        - ⛰️ Elevation
        - 💧 Water bodies
        """)

# Prediction History page
def prediction_history_page():
    """Display prediction history"""
    st.markdown("### 📊 Prediction History")
    
    history = get_user_history(st.session_state.username)
    
    if not history:
        st.info("No prediction history yet. Make some predictions on the Map Predictor page!")
    else:
        # Display history in reverse chronological order
        for i, pred in enumerate(reversed(history[-20:])):  # Show last 20
            with st.expander(f"Prediction {len(history) - i} - {pred['timestamp'][:19]}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Handle both dict (map) and string (personal entry) location formats
                    if isinstance(pred['location'], dict):
                        st.write(f"**Location:** {pred['location']['lat']:.4f}, {pred['location']['lon']:.4f}")
                    else:
                        st.write(f"**Source:** {pred['location']}")
                    st.write(f"**Risk Level:** {pred['risk_level']}")
                    st.write(f"**Confidence:** {pred['confidence']*100:.1f}%")
                
                with col2:
                    st.write("**Features:**")
                    if isinstance(pred['features'], dict):
                        for key, value in pred['features'].items():
                            st.write(f"- {key}: {value}")
                    else:
                        st.write(str(pred['features']))

# Account Info page
def account_info_page():
    """Display account information"""
    st.markdown("### 👤 Account Information")
    
    users = load_users()
    user_info = users.get(st.session_state.username, {})
    
    st.write(f"**Username:** {st.session_state.username}")
    st.write(f"**Account Created:** {user_info.get('created_at', 'N/A')[:10]}")
    
    history = get_user_history(st.session_state.username)
    st.write(f"**Total Predictions:** {len(history)}")
    
    st.markdown("---")
    st.markdown("### Change Password")
    
    old_password = st.text_input("Current Password", type="password", key="old_pass")
    new_password = st.text_input("New Password", type="password", key="new_pass")
    confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_new_pass")
    
    if st.button("Update Password", key="update_pass_btn"):
        if old_password and new_password and confirm_password:
            users = load_users()
            if users[st.session_state.username]["password"] == hash_password(old_password):
                if new_password == confirm_password:
                    if len(new_password) >= 6:
                        users[st.session_state.username]["password"] = hash_password(new_password)
                        save_users(users)
                        st.success("Password updated successfully!")
                    else:
                        st.error("Password must be at least 6 characters long")
                else:
                    st.error("New passwords do not match")
            else:
                st.error("Current password is incorrect")
        else:
            st.warning("Please fill in all fields")

# Main application
def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        render_sidebar()
        
        if st.session_state.current_page == "Personal Entry":
            personal_entry_page()
        elif st.session_state.current_page == "Map Predictor":
            map_predictor_page()
        elif st.session_state.current_page == "Prediction History":
            prediction_history_page()
        elif st.session_state.current_page == "Account Info":
            account_info_page()

if __name__ == "__main__":
    main()
