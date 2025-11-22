# Malaria Risk Predictor Application

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

# Page configuration
st.set_page_config(
    page_title="Malaria Risk Predictor",
    page_icon="🦟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    h1 {
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #FF6B35;
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
    .stNumberInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
    }
    .risk-high {
        color: #FF6B35;
        font-size: 2rem;
        font-weight: bold;
    }
    .risk-medium {
        color: #FFA500;
        font-size: 2rem;
        font-weight: bold;
    }
    .risk-low {
        color: #4CAF50;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# File paths
USERS_FILE = "users.json"
MODEL_FILE = "malaria_model.pkl"

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "malaria_model" not in st.session_state:
    st.session_state.malaria_model = None

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

# Malaria prediction model
def create_malaria_model():
    """Create and train a malaria risk prediction model"""
    # Generate synthetic training data based on malaria risk factors
    np.random.seed(42)
    n_samples = 1000
    
    # Features: age, temperature, recent_travel, mosquito_exposure, symptoms, 
    #           preventive_medication, region_risk, immune_status
    data = []
    for _ in range(n_samples):
        age = np.random.randint(1, 80)
        temperature = np.random.normal(37.0, 1.0)
        recent_travel = np.random.choice([0, 1], p=[0.7, 0.3])
        mosquito_exposure = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])  # 0=low, 1=medium, 2=high
        symptoms = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.15, 0.05])  # number of symptoms
        preventive_medication = np.random.choice([0, 1], p=[0.6, 0.4])
        region_risk = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])  # 0=low, 1=medium, 2=high
        immune_status = np.random.choice([0, 1], p=[0.7, 0.3])  # 0=no immunity, 1=some immunity
        
        # Calculate risk score (higher = more likely to have malaria)
        risk_score = (
            (temperature - 37.0) * 2 +  # Fever increases risk
            recent_travel * 3 +
            mosquito_exposure * 2 +
            symptoms * 2 +
            (1 - preventive_medication) * 2 +
            region_risk * 2 +
            (1 - immune_status) * 1.5
        )
        
        # Convert to binary label (1 = high risk, 0 = low risk)
        label = 1 if risk_score > 8 else 0
        
        data.append([
            age, temperature, recent_travel, mosquito_exposure, 
            symptoms, preventive_medication, region_risk, immune_status, label
        ])
    
    df = pd.DataFrame(data, columns=[
        "age", "temperature", "recent_travel", "mosquito_exposure",
        "symptoms", "preventive_medication", "region_risk", "immune_status", "risk"
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

def predict_malaria_risk(age, temperature, recent_travel, mosquito_exposure, 
                        symptoms, preventive_medication, region_risk, immune_status):
    """Predict malaria risk based on input features"""
    model = load_or_create_model()
    
    features = np.array([[
        age, temperature, recent_travel, mosquito_exposure,
        symptoms, preventive_medication, region_risk, immune_status
    ]])
    
    risk_probability = model.predict_proba(features)[0][1]
    risk_class = model.predict(features)[0]
    
    return risk_class, risk_probability

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability >= 0.7:
        return "High", "🔴"
    elif probability >= 0.4:
        return "Medium", "🟡"
    else:
        return "Low", "🟢"

# Main application
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

def prediction_page():
    """Display prediction page"""
    st.markdown("<h1 style='text-align: center; color: white;'>🦟 Malaria Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: white;'>Welcome, <b>{st.session_state.username}</b>!</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 style='color: white;'>Patient Information</h3>", unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=100, value=30, key="age")
        temperature = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1, key="temp")
        recent_travel = st.selectbox("Recent Travel to Malaria-Endemic Area", 
                                   ["No", "Yes"], key="travel")
        mosquito_exposure = st.selectbox("Mosquito Exposure Level", 
                                        ["Low", "Medium", "High"], key="mosquito")
        symptoms = st.number_input("Number of Symptoms (fever, chills, headache, etc.)", 
                                  min_value=0, max_value=10, value=0, key="symptoms")
    
    with col2:
        st.markdown("<h3 style='color: white;'>Additional Factors</h3>", unsafe_allow_html=True)
        preventive_medication = st.selectbox("Taking Preventive Medication", 
                                           ["No", "Yes"], key="medication")
        region_risk = st.selectbox("Current Region Risk Level", 
                                  ["Low", "Medium", "High"], key="region")
        immune_status = st.selectbox("Immune Status", 
                                   ["No Previous Exposure", "Some Immunity"], key="immune")
    
    if st.button("Predict Malaria Risk", key="predict_btn"):
        # Convert inputs to model format
        recent_travel_val = 1 if recent_travel == "Yes" else 0
        mosquito_exposure_map = {"Low": 0, "Medium": 1, "High": 2}
        mosquito_exposure_val = mosquito_exposure_map[mosquito_exposure]
        preventive_medication_val = 1 if preventive_medication == "Yes" else 0
        region_risk_map = {"Low": 0, "Medium": 1, "High": 2}
        region_risk_val = region_risk_map[region_risk]
        immune_status_val = 1 if immune_status == "Some Immunity" else 0
        
        risk_class, risk_probability = predict_malaria_risk(
            age, temperature, recent_travel_val, mosquito_exposure_val,
            symptoms, preventive_medication_val, region_risk_val, immune_status_val
        )
        
        risk_level, risk_emoji = get_risk_level(risk_probability)
        
        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style='color: white;'>Prediction Result</h2>
            <p class="risk-{risk_level.lower()}" style='font-size: 3rem;'>{risk_emoji}</p>
            <h3 class="risk-{risk_level.lower()}">{risk_level} Risk</h3>
            <p style='color: white; font-size: 1.2rem;'>Risk Probability: <b>{(risk_probability * 100):.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### Recommendations")
        if risk_level == "High":
            st.warning("⚠️ **Immediate Action Recommended:** Please consult a healthcare professional immediately. Consider getting a malaria test.")
        elif risk_level == "Medium":
            st.info("ℹ️ **Precautionary Measures:** Monitor symptoms closely. Consider preventive measures and consult a doctor if symptoms worsen.")
        else:
            st.success("✅ **Low Risk:** Continue preventive measures. Stay vigilant about mosquito protection.")
        
        # Additional information
        with st.expander("View Detailed Analysis"):
            st.write(f"**Age:** {age} years")
            st.write(f"**Temperature:** {temperature}°C")
            st.write(f"**Recent Travel:** {recent_travel}")
            st.write(f"**Mosquito Exposure:** {mosquito_exposure}")
            st.write(f"**Symptoms Count:** {symptoms}")
            st.write(f"**Preventive Medication:** {preventive_medication}")
            st.write(f"**Region Risk:** {region_risk}")
            st.write(f"**Immune Status:** {immune_status}")
            st.write(f"**Risk Score:** {risk_probability:.3f}")
    
    # Logout button
    st.markdown("---")
    if st.button("Logout", key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# Main app logic
def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        prediction_page()

if __name__ == "__main__":
    main()
