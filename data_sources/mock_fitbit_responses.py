"""
Fixtures representing Fitbit API responses for testing purposes.

These structures emulate payloads returned by Fitbit endpoints to avoid
direct API calls during automated testing.
"""

from __future__ import annotations

from datetime import datetime

AUTH_TOKEN_RESPONSE = {
    "access_token": "ACCESS_TOKEN",
    "refresh_token": "REFRESH_TOKEN",
    "expires_in": 3600,
    "scope": "activity heartrate sleep",
    "user_id": "12345",
}

HEART_RATE_INTRADAY_RESPONSE = {
    "activities-heart-intraday": {
        "dataset": [
            {"time": "00:00:00", "value": 65},
            {"time": "00:01:00", "value": 66},
            {"time": "00:02:00", "value": 70},
        ]
    }
}

ACTIVITY_SUMMARY_RESPONSE = {
    "summary": {
        "steps": 5000,
        "caloriesOut": 2200,
        "floors": 8,
        "fairlyActiveMinutes": 30,
        "veryActiveMinutes": 15,
    }
}

SLEEP_LOGS_RESPONSE = {
    "sleep": [
        {
            "dateOfSleep": datetime.utcnow().strftime("%Y-%m-%d"),
            "duration": 25200000,
            "minutesAsleep": 360,
            "minutesAwake": 60,
            "levels": {
                "data": [
                    {"dateTime": "00:00:00", "level": "light", "seconds": 1800},
                    {"dateTime": "00:30:00", "level": "deep", "seconds": 1200},
                ]
            },
        }
    ]
}

