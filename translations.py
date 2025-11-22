"""
Multi-language translations for MalariaAI
Supports English, Kiswahili, and other local languages
"""

TRANSLATIONS = {
    'en': {
        # Authentication
        'title': 'MalariaAI - Kenya Risk Dashboard',
        'subtitle': 'Real-time malaria risk monitoring across Kenya',
        'login': 'Login',
        'register': 'Register',
        'username': 'Username',
        'password': 'Password',
        'email': 'Email',
        'logout': 'Logout',
        
        # User Profile
        'personal_info': 'Personal Information',
        'name': 'Full Name',
        'age': 'Your Age',
        'gender': 'Gender',
        'county': 'Your County',
        
        # Health Assessment
        'health_assessment': 'Health Assessment',
        'temperature': 'Current Body Temperature (°C)',
        'travel': 'Have you traveled to malaria-endemic areas recently?',
        'mosquito_exposure': 'How would you rate your mosquito exposure?',
        'symptoms': 'Select any symptoms you\'re experiencing:',
        'medication': 'Are you taking antimalarial medication?',
        'bed_nets': 'Do you use bed nets?',
        'repellent': 'Do you use insect repellent?',
        'immune_status': 'How would you describe your immune system?',
        
        # Symptoms
        'fever': 'Fever',
        'headache': 'Headache', 
        'chills': 'Chills',
        'nausea': 'Nausea',
        'vomiting': 'Vomiting',
        'fatigue': 'Fatigue',
        'muscle_aches': 'Muscle aches',
        'sweating': 'Sweating',
        
        # Risk Assessment
        'predict': 'Get My Risk Assessment',
        'risk_assessment': 'Risk Assessment',
        'high_risk': 'HIGH RISK',
        'medium_risk': 'MEDIUM RISK',
        'low_risk': 'LOW RISK',
        'recommendation': 'Recommendation',
        'risk_factors': 'Your Risk Factors',
        'protective_factors': 'Protective Factors',
        'next_steps': 'Recommended Next Steps',
        
        # Recommendations
        'seek_immediate_care': 'Seek immediate medical attention',
        'monitor_symptoms': 'Monitor symptoms closely',
        'continue_prevention': 'Continue preventive measures',
        'visit_healthcare': 'Visit nearest healthcare facility',
        'blood_test': 'Request malaria blood test',
        'monitor_temperature': 'Keep monitoring your temperature',
        'stay_hydrated': 'Stay hydrated and rest',
        
        # General
        'yes': 'Yes',
        'no': 'No',
        'male': 'Male',
        'female': 'Female',
        'other': 'Other',
        'normal': 'Normal',
        'strong': 'Strong',
        'compromised': 'Compromised',
        'online': 'Online',
        'offline': 'Offline Mode',
        'sync_data': 'Sync Offline Data',
        'offline_data': 'Offline Data'
    },
    
    'sw': {
        # Authentication
        'title': 'MalariaAI - Dashibodi ya Hatari Kenya',
        'subtitle': 'Ufuatiliaji wa hatari ya malaria wakati halisi nchini Kenya',
        'login': 'Ingia',
        'register': 'Jisajili',
        'username': 'Jina la mtumiaji',
        'password': 'Nywila',
        'email': 'Barua pepe',
        'logout': 'Toka',
        
        # User Profile
        'personal_info': 'Maelezo ya Kibinafsi',
        'name': 'Jina kamili',
        'age': 'Umri wako',
        'gender': 'Jinsia',
        'county': 'Kaunti yako',
        
        # Health Assessment
        'health_assessment': 'Tathmini ya Afya',
        'temperature': 'Joto la mwili sasa (°C)',
        'travel': 'Je, umesafiri kwenye maeneo ya malaria hivi karibuni?',
        'mosquito_exposure': 'Unaonaje kiwango cha mbu unaokutana nacho?',
        'symptoms': 'Chagua dalili zozote unazohisi:',
        'medication': 'Je, unatumia dawa za kuzuia malaria?',
        'bed_nets': 'Je, unatumia chandarua?',
        'repellent': 'Je, unatumia dawa za kufukuza wadudu?',
        'immune_status': 'Unaonaje mfumo wako wa kinga?',
        
        # Symptoms
        'fever': 'Homa',
        'headache': 'Maumivu ya kichwa',
        'chills': 'Baridi',
        'nausea': 'Kichefuchefu',
        'vomiting': 'Kutapika',
        'fatigue': 'Uchovu',
        'muscle_aches': 'Maumivu ya misuli',
        'sweating': 'Jasho',
        
        # Risk Assessment
        'predict': 'Pata Tathmini ya Hatari Yangu',
        'risk_assessment': 'Tathmini ya Hatari',
        'high_risk': 'HATARI KUBWA',
        'medium_risk': 'HATARI YA KATI',
        'low_risk': 'HATARI NDOGO',
        'recommendation': 'Mapendekezo',
        'risk_factors': 'Sababu za Hatari Yako',
        'protective_factors': 'Mambo Yanayokulinda',
        'next_steps': 'Hatua Zinazofuata',
        
        # Recommendations
        'seek_immediate_care': 'Tafuta msaada wa matibabu mara moja',
        'monitor_symptoms': 'Fuatilia dalili kwa karibu',
        'continue_prevention': 'Endelea na njia za kujikinga',
        'visit_healthcare': 'Tembelea kituo cha afya karibu zaidi',
        'blood_test': 'Omba upimaji wa damu wa malaria',
        'monitor_temperature': 'Endelea kupima joto lako',
        'stay_hydrated': 'Kunywa maji mengi na pumzika',
        
        # General
        'yes': 'Ndiyo',
        'no': 'Hapana',
        'male': 'Mwanaume',
        'female': 'Mwanamke',
        'other': 'Nyingine',
        'normal': 'Kawaida',
        'strong': 'Imara',
        'compromised': 'Dhaifu',
        'online': 'Mtandaoni',
        'offline': 'Nje ya Mtandao',
        'sync_data': 'Sawazisha Data ya Nje ya Mtandao',
        'offline_data': 'Data ya Nje ya Mtandao'
    }
}

def get_text(key, language='en'):
    """Get translated text for given key and language"""
    return TRANSLATIONS.get(language, {}).get(key, key)

def get_available_languages():
    """Get list of available languages"""
    return {
        'English': 'en',
        'Kiswahili': 'sw'
    }

def get_symptom_translations(language='en'):
    """Get symptom translations with icons for visual accessibility"""
    symptom_icons = {
        'fever': '🤒',
        'headache': '🤕', 
        'chills': '🥶',
        'nausea': '🤢',
        'vomiting': '🤮',
        'fatigue': '😴',
        'muscle_aches': '💪',
        'sweating': '💦'
    }
    
    symptoms = {}
    for key, icon in symptom_icons.items():
        text = get_text(key, language)
        symptoms[key] = f"{icon} {text}"
    
    return symptoms

def get_risk_level_color(risk_level):
    """Get color coding for risk levels"""
    colors = {
        'HIGH': '#ff4757',
        'MEDIUM': '#ffa500', 
        'LOW': '#00a676'
    }
    
    for level, color in colors.items():
        if level in risk_level.upper():
            return color
    
    return '#666666'

def get_risk_level_emoji(risk_level):
    """Get emoji for risk levels"""
    emojis = {
        'HIGH': '🔴',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    
    for level, emoji in emojis.items():
        if level in risk_level.upper():
            return emoji
    
    return '⚪'