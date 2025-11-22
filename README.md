# MalariaAI - Kenya Risk Dashboard

A Streamlit web application for predicting malaria risk with focus on Kenya, featuring an interactive map and real-time risk monitoring.

## Features

- **Kenya-Focused Map**: Interactive map showing malaria risk zones across Kenya
- **AI Risk Prediction**: Machine learning-powered malaria risk assessment
- **Real-time Monitoring**: Live climate data and case statistics
- **Risk Alerts**: Active alerts for high-risk areas
- **Clean Interface**: No charts, focused on map visualization

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run streamlit_app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Usage

1. **View Kenya Map**: Interactive map showing risk zones in major Kenyan cities
2. **Enter Patient Data**: Use sidebar form to input health information
3. **Get Risk Assessment**: Receive instant risk prediction with recommendations
4. **Monitor Alerts**: View active alerts for different regions

## Kenya Cities Covered

- **Nairobi**: High risk - 156 cases
- **Mombasa**: Medium risk - 89 cases
- **Eldoret**: High risk - 67 cases
- **Kisumu**: Medium risk - 45 cases
- **Meru**: Low risk - 23 cases
- **Malindi**: High risk - 78 cases

## Model Details

Random Forest Classifier considering:
- Patient demographics and symptoms
- Environmental factors
- Travel history and preventive measures

## File Structure

```
Medical_app/
├── streamlit_app.py     # Main Streamlit application
├── requirements.txt     # Python dependencies
├── malaria_model.pkl   # ML model (auto-generated)
├── users.json         # User data (auto-generated)
└── README.md          # Documentation
```

## License

Educational and research purposes.

