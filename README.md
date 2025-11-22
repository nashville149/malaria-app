# Malaria Risk Predictor

A web application for predicting malaria risk based on patient information, travel history, symptoms, and environmental factors.

## Features

- **User Authentication**: Secure login and registration system
- **Risk Prediction**: AI-powered malaria risk assessment using machine learning
- **Comprehensive Input**: Collects age, temperature, travel history, mosquito exposure, symptoms, medication status, region risk, and immune status
- **Risk Classification**: Provides High, Medium, or Low risk predictions with probability scores
- **Recommendations**: Actionable recommendations based on risk level

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run index.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Usage

1. **Register/Login**: Create a new account or login with existing credentials
2. **Enter Patient Information**: Fill in the required fields:
   - Age
   - Body Temperature
   - Recent Travel to Malaria-Endemic Areas
   - Mosquito Exposure Level
   - Number of Symptoms
   - Preventive Medication Status
   - Current Region Risk Level
   - Immune Status
3. **Get Prediction**: Click "Predict Malaria Risk" to get the risk assessment
4. **Review Recommendations**: Follow the recommendations based on your risk level

## Model Details

The application uses a Random Forest Classifier trained on synthetic data that considers:
- Patient demographics (age)
- Clinical symptoms (temperature, symptom count)
- Travel history
- Environmental factors (mosquito exposure, region risk)
- Preventive measures (medication, immune status)

## File Structure

```
Medical_app/
├── index.py              # Main application file
├── requirements.txt      # Python dependencies
├── users.json           # User database (created automatically)
├── malaria_model.pkl    # Trained model (created automatically)
├── Dockerfile           # Docker configuration
└── deployment/          # Deployment documentation
```

## Security Notes

- Passwords are hashed using SHA256
- User data is stored locally in JSON format
- For production use, consider implementing a proper database and stronger encryption

## Deployment

See `deployment/azure/README.md` for Azure deployment instructions.

## License

This project is for educational and research purposes.

