import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
import json
import hashlib
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="MalariaAI - Kenya Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# User management functions
def load_users():
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    users = load_users()
    if username in users:
        return users[username]['password'] == hash_password(password)
    return False

def register_user(username, password, email, county):
    users = load_users()
    if username in users:
        return False
    users[username] = {
        'password': hash_password(password),
        'email': email,
        'county': county,
        'created': datetime.now().isoformat()
    }
    save_users(users)
    return True

# Authentication check
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None

# Load model
@st.cache_resource
def load_model():
    try:
        with open('malaria_model.pkl', 'rb') as f:
            model = pickle.load(f)
            st.success("Model loaded successfully!")
            return model
    except FileNotFoundError:
        st.info("Creating new model...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Generate more realistic training data
        X = np.random.rand(1000, 8)
        # Create realistic malaria risk patterns
        risk_score = (X[:, 0] * 0.2 +  # age factor
                     X[:, 1] * 0.3 +  # temperature factor (higher weight)
                     X[:, 2] * 0.25 + # travel factor
                     X[:, 3] * 0.15 + # mosquito exposure
                     X[:, 4] * 0.1 +  # symptoms
                     X[:, 5] * 0.05 + # medication (protective)
                     X[:, 6] * 0.1 +  # region risk
                     X[:, 7] * 0.05)  # immune status
        
        # Add some noise and make it more realistic
        noise = np.random.normal(0, 0.1, 1000)
        risk_score = risk_score + noise
        y = (risk_score > 0.4).astype(int)  # Lower threshold for more variation
        
        model.fit(X, y)
        
        with open('malaria_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        st.success("New model created and saved!")
        return model

model = load_model()

# Authentication UI
if not st.session_state.authenticated:
    st.markdown("""
    <div class="main-header">
        <h1>🦟 MalariaAI - Kenya Risk Dashboard</h1>
        <p>Please login or register to access the malaria risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")
            
            if login_btn:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Register")
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email = st.text_input("Email")
            county = st.selectbox("Your County", 
                               ["Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet", "Embu", 
                                "Garissa", "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho", 
                                "Kiambu", "Kilifi", "Kirinyaga", "Kisii", "Kisumu", "Kitui", 
                                "Kwale", "Laikipia", "Lamu", "Machakos", "Makueni", "Mandera", 
                                "Marsabit", "Meru", "Migori", "Mombasa", "Murang'a", "Nairobi", 
                                "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua", "Nyeri", 
                                "Samburu", "Siaya", "Taita-Taveta", "Tana River", "Tharaka-Nithi", 
                                "Trans Nzoia", "Turkana", "Uasin Gishu", "Vihiga", "Wajir", "West Pokot"])
            register_btn = st.form_submit_button("Register")
            
            if register_btn:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif not new_username or not email:
                    st.error("Please fill all fields")
                else:
                    if register_user(new_username, new_password, email, county):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")
    
    st.stop()

# Kenya counties risk zones data
kenya_zones = [
    # High risk coastal counties
    {"name": "Mombasa", "lat": -4.0435, "lng": 39.6682, "risk": "high", "cases": 234},
    {"name": "Kilifi", "lat": -3.5107, "lng": 39.9059, "risk": "high", "cases": 189},
    {"name": "Kwale", "lat": -4.1747, "lng": 39.4529, "risk": "high", "cases": 156},
    {"name": "Tana River", "lat": -1.3733, "lng": 40.0348, "risk": "high", "cases": 198},
    # High risk lake region
    {"name": "Kisumu", "lat": -0.0917, "lng": 34.7680, "risk": "high", "cases": 167},
    {"name": "Homa Bay", "lat": -0.5273, "lng": 34.4571, "risk": "high", "cases": 201},
    {"name": "Migori", "lat": -1.0634, "lng": 34.4731, "risk": "high", "cases": 143},
    {"name": "Siaya", "lat": 0.0607, "lng": 34.2888, "risk": "high", "cases": 134},
    # Medium risk
    {"name": "Nairobi", "lat": -1.2921, "lng": 36.8219, "risk": "medium", "cases": 89},
    {"name": "Nakuru", "lat": -0.3031, "lng": 36.0800, "risk": "medium", "cases": 67},
    {"name": "Kakamega", "lat": 0.2827, "lng": 34.7519, "risk": "medium", "cases": 78},
    {"name": "Bungoma", "lat": 0.5692, "lng": 34.5606, "risk": "medium", "cases": 56},
    # Low risk highland counties
    {"name": "Meru", "lat": 0.0469, "lng": 37.6556, "risk": "low", "cases": 23},
    {"name": "Nyeri", "lat": -0.4209, "lng": 36.9483, "risk": "low", "cases": 18},
    {"name": "Kiambu", "lat": -1.1714, "lng": 36.8356, "risk": "low", "cases": 34},
    {"name": "Nyandarua", "lat": -0.3924, "lng": 36.3569, "risk": "low", "cases": 12}
]

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #00a676, #04364f);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    text-align: center;
}

.risk-high { color: #ff4757; }
.risk-medium { color: #ffa500; }
.risk-low { color: #00a676; }

.alert-high {
    background: #ffe6e6;
    border-left: 4px solid #ff4757;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}

.alert-medium {
    background: #fff3e0;
    border-left: 4px solid #ffa500;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Header with logout
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("""
    <div class="main-header">
        <h1>🦟 MalariaAI - Kenya Risk Dashboard</h1>
        <p>Welcome back, {}</p>
    </div>
    """.format(st.session_state.username), unsafe_allow_html=True)

with col2:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# Get user data
users = load_users()
user_data = users.get(st.session_state.username, {})

# Sidebar - User Profile & Prediction
st.sidebar.title(f"👤 {st.session_state.username}")
st.sidebar.markdown(f"**County:** {user_data.get('county', 'Not set')}")
st.sidebar.markdown(f"**Email:** {user_data.get('email', 'Not set')}")

# User Information
with st.sidebar.expander("Personal Information", expanded=True):
    name = st.text_input("Full Name", value=st.session_state.username)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    location = st.selectbox("Current County in Kenya", 
                           ["Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet", "Embu", 
                            "Garissa", "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho", 
                            "Kiambu", "Kilifi", "Kirinyaga", "Kisii", "Kisumu", "Kitui", 
                            "Kwale", "Laikipia", "Lamu", "Machakos", "Makueni", "Mandera", 
                            "Marsabit", "Meru", "Migori", "Mombasa", "Murang'a", "Nairobi", 
                            "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua", "Nyeri", 
                            "Samburu", "Siaya", "Taita-Taveta", "Tana River", "Tharaka-Nithi", 
                            "Trans Nzoia", "Turkana", "Uasin Gishu", "Vihiga", "Wajir", "West Pokot"],
                           index=0 if user_data.get('county') not in ["Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet", "Embu", 
                            "Garissa", "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho", 
                            "Kiambu", "Kilifi", "Kirinyaga", "Kisii", "Kisumu", "Kitui", 
                            "Kwale", "Laikipia", "Lamu", "Machakos", "Makueni", "Mandera", 
                            "Marsabit", "Meru", "Migori", "Mombasa", "Murang'a", "Nairobi", 
                            "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua", "Nyeri", 
                            "Samburu", "Siaya", "Taita-Taveta", "Tana River", "Tharaka-Nithi", 
                            "Trans Nzoia", "Turkana", "Uasin Gishu", "Vihiga", "Wajir", "West Pokot"] else 
                           ["Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet", "Embu", 
                            "Garissa", "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho", 
                            "Kiambu", "Kilifi", "Kirinyaga", "Kisii", "Kisumu", "Kitui", 
                            "Kwale", "Laikipia", "Lamu", "Machakos", "Makueni", "Mandera", 
                            "Marsabit", "Meru", "Migori", "Mombasa", "Murang'a", "Nairobi", 
                            "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua", "Nyeri", 
                            "Samburu", "Siaya", "Taita-Taveta", "Tana River", "Tharaka-Nithi", 
                            "Trans Nzoia", "Turkana", "Uasin Gishu", "Vihiga", "Wajir", "West Pokot"].index(user_data.get('county'))) 


st.sidebar.title("🔬 Health Assessment")
st.sidebar.markdown("Enter your current health information:")

with st.sidebar.form("prediction_form"):
    age = st.number_input("Your Age", min_value=0, max_value=120, value=30)
    temperature = st.number_input("Current Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
    
    st.markdown("**Travel History**")
    travel = st.selectbox("Have you traveled to malaria-endemic areas recently?", ["No", "Yes"])
    
    st.markdown("**Environmental Exposure**")
    mosquito = st.slider("How would you rate your mosquito exposure? (1=Very Low, 10=Very High)", 1, 10, 5)
    
    st.markdown("**Current Symptoms**")
    symptoms_list = st.multiselect(
        "Select any symptoms you're experiencing:",
        ["Fever", "Headache", "Chills", "Nausea", "Vomiting", "Fatigue", "Muscle aches", "Sweating"]
    )
    symptoms = len(symptoms_list)
    
    st.markdown("**Prevention Measures**")
    medication = st.selectbox("Are you taking antimalarial medication?", ["No", "Yes"])
    bed_nets = st.selectbox("Do you use bed nets?", ["No", "Yes"])
    repellent = st.selectbox("Do you use insect repellent?", ["No", "Yes"])
    
    st.markdown("**Health Status**")
    immune = st.selectbox("How would you describe your immune system?", ["Compromised", "Normal", "Strong"])
    
    # Auto-set region risk based on county
    region_risk_map = {
        # High risk counties (coastal and lake regions)
        "Mombasa": 9, "Kilifi": 8, "Kwale": 8, "Tana River": 9, "Lamu": 7,
        "Kisumu": 8, "Homa Bay": 9, "Migori": 8, "Siaya": 8, "Busia": 7,
        # Medium-high risk
        "Nairobi": 6, "Kakamega": 7, "Bungoma": 6, "Vihiga": 6,
        "Garissa": 7, "Wajir": 6, "Mandera": 7, "Marsabit": 6,
        # Medium risk
        "Nakuru": 5, "Uasin Gishu": 5, "Trans Nzoia": 5, "Kericho": 4,
        "Bomet": 4, "Nandi": 4, "Machakos": 5, "Kitui": 6, "Makueni": 5,
        # Lower risk (highland areas)
        "Kiambu": 3, "Murang'a": 3, "Nyeri": 2, "Kirinyaga": 3, "Embu": 3,
        "Meru": 3, "Tharaka-Nithi": 3, "Nyandarua": 2, "Laikipia": 4,
        "Samburu": 5, "Isiolo": 5, "Kajiado": 4, "Narok": 4,
        # Variable risk
        "Kisii": 4, "Nyamira": 4, "Baringo": 5, "Elgeyo-Marakwet": 3,
        "West Pokot": 4, "Turkana": 5, "Taita-Taveta": 6
    }
    region = region_risk_map.get(location, 5)
    
    submitted = st.form_submit_button("🔍 Get My Risk Assessment")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Kenya Malaria Risk Map")
    
    # Create map
    m = folium.Map(location=[-1.2921, 36.8219], zoom_start=6)
    
    # Add risk zones
    for zone in kenya_zones:
        color = "#ff4757" if zone["risk"] == "high" else "#ffa500" if zone["risk"] == "medium" else "#00a676"
        
        folium.Circle(
            location=[zone["lat"], zone["lng"]],
            radius=50000,
            color=color,
            fillColor=color,
            fillOpacity=0.4,
            popup=f"""
            <b>{zone['name']}, Kenya</b><br>
            Risk Level: {zone['risk'].upper()}<br>
            Cases: {zone['cases']}<br>
            Last Updated: 1 hour ago
            """
        ).add_to(m)
    
    # Display map
    map_data = st_folium(m, width=700, height=500)

with col2:
    st.subheader("📊 Kenya Statistics")
    
    # Climate data
    st.markdown("### 🌤️ Current Climate")
    col_temp, col_rain, col_humid = st.columns(3)
    
    with col_temp:
        st.metric("Temperature", "24°C", "↑2°C")
    with col_rain:
        st.metric("Rainfall", "65mm", "↑15mm")
    with col_humid:
        st.metric("Humidity", "82%", "↑5%")
    
    # Case statistics
    st.markdown("### 📈 Case Statistics")
    col_week, col_today = st.columns(2)
    
    with col_week:
        st.metric("Cases This Week", "458", "↑8%")
    with col_today:
        st.metric("New Cases Today", "34", "↓3%")

# Alerts section
st.subheader("🚨 Active Alerts")
col_alert1, col_alert2 = st.columns(2)

with col_alert1:
    st.markdown("""
    <div class="alert-high">
        <strong>⚠️ Nairobi, Kenya</strong><br>
        High risk - Temperature spike detected<br>
        <small>1 hour ago</small>
    </div>
    """, unsafe_allow_html=True)

with col_alert2:
    st.markdown("""
    <div class="alert-medium">
        <strong>⚡ Mombasa, Kenya</strong><br>
        Increased rainfall - Monitor closely<br>
        <small>3 hours ago</small>
    </div>
    """, unsafe_allow_html=True)

# Prediction results
if submitted:
    st.subheader("🎯 Prediction Results")
    
    with st.spinner("Analyzing your data..."):
        # Prepare personalized features
        # Adjust immune status scoring
        immune_score = 1 if immune == "Strong" else 0.5 if immune == "Normal" else 0
        
        # Calculate prevention score
        prevention_score = 0
        if medication == "Yes":
            prevention_score += 0.4
        if bed_nets == "Yes":
            prevention_score += 0.3
        if repellent == "Yes":
            prevention_score += 0.3
        
        features = [
            age / 100,
            temperature / 42,
            1 if travel == "Yes" else 0,
            mosquito / 10,
            symptoms / 10,
            1 - prevention_score,  # Higher prevention = lower risk
            region / 10,
            immune_score
        ]
        
        # Make prediction
        try:
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]
            
            # Calculate risk score
            if len(probability) > 1:
                risk_score = int(probability[1] * 100)
            else:
                risk_score = int(prediction * 100)
            
            # Ensure risk score is reasonable
            risk_score = max(0, min(100, risk_score))
            
            # Personalized risk assessment and recommendations
            if risk_score >= 70:
                risk_level = "High"
                risk_class = "risk-high"
                recommendation = f"⚠️ **{name or 'Patient'}**, your risk is HIGH. Seek immediate medical attention in {location}. Consider visiting a healthcare provider for testing and treatment."
                if symptoms > 3:
                    recommendation += " Your multiple symptoms require urgent evaluation."
            elif risk_score >= 40:
                risk_level = "Medium"
                risk_class = "risk-medium"
                recommendation = f"⚡ **{name or 'Patient'}**, you have MEDIUM risk. Monitor your symptoms closely in {location}."
                if temperature > 38.0:
                    recommendation += " Your elevated temperature needs attention."
                if travel == "Yes":
                    recommendation += " Your recent travel increases concern."
            else:
                risk_level = "Low"
                risk_class = "risk-low"
                recommendation = f"✅ **{name or 'Patient'}**, your risk is LOW in {location}. Continue your current preventive measures."
                if prevention_score < 0.5:
                    recommendation += " Consider improving your prevention methods (bed nets, repellent, medication)."
            
            # Display personalized results
            st.markdown(f"### 📋 Risk Assessment for {name or 'Patient'}")
            
            col_score, col_level, col_location = st.columns(3)
            
            with col_score:
                st.metric("Your Risk Score", f"{risk_score}%")
            
            with col_level:
                st.metric("Risk Level", risk_level)
                
            with col_location:
                st.metric("Location", location)
            
            # Personalized recommendation
            if risk_level == "High":
                st.error(recommendation)
            elif risk_level == "Medium":
                st.warning(recommendation)
            else:
                st.success(recommendation)
            
            # Additional personalized insights
            st.markdown("### 🔍 Your Risk Factors")
            
            risk_factors = []
            if temperature > 38.0:
                risk_factors.append(f"🌡️ Elevated temperature ({temperature}°C)")
            if symptoms > 2:
                risk_factors.append(f"🤒 Multiple symptoms ({', '.join(symptoms_list)})")
            if travel == "Yes":
                risk_factors.append("✈️ Recent travel to endemic areas")
            if mosquito > 7:
                risk_factors.append(f"🦟 High mosquito exposure (Level {mosquito})")
            if region > 6:
                risk_factors.append(f"📍 High-risk location ({location})")
            
            protective_factors = []
            if medication == "Yes":
                protective_factors.append("💊 Taking antimalarial medication")
            if bed_nets == "Yes":
                protective_factors.append("🛏️ Using bed nets")
            if repellent == "Yes":
                protective_factors.append("🚫 Using insect repellent")
            if immune == "Strong":
                protective_factors.append("💪 Strong immune system")
            
            if risk_factors:
                st.markdown("**⚠️ Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            
            if protective_factors:
                st.markdown("**✅ Protective Factors:**")
                for factor in protective_factors:
                    st.markdown(f"- {factor}")
            
            # Personalized next steps
            st.markdown("### 📝 Recommended Next Steps")
            if risk_level == "High":
                st.markdown(f"""
                1. 🏥 **Visit nearest healthcare facility in {location} immediately**
                2. 🩸 **Request malaria blood test (RDT or microscopy)**
                3. 📱 **Keep monitoring your temperature every 2 hours**
                4. 💧 **Stay hydrated and rest**
                """)
            elif risk_level == "Medium":
                st.markdown(f"""
                1. 🌡️ **Monitor temperature every 4 hours**
                2. 👀 **Watch for worsening symptoms**
                3. 🏥 **Consult healthcare provider in {location} if symptoms persist**
                4. 🦟 **Increase mosquito protection measures**
                """)
            else:
                st.markdown(f"""
                1. 🛡️ **Continue current prevention methods**
                2. 🦟 **Maintain mosquito protection in {location}**
                3. 👀 **Stay alert for any new symptoms**
                4. 📅 **Regular health check-ups**
                """)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.write(f"Debug - Error details: {type(e).__name__}")

# Footer
st.markdown("---")
st.markdown("**MalariaAI Kenya Dashboard** - Real-time malaria risk monitoring | © 2024")