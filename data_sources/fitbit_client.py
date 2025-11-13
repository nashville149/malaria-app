"""
Fitbit Web API client for ingesting wearable telemetry.

This module manages OAuth 2.0 authentication flows and provides helper
methods to fetch heart rate, sleep, and activity data in a normalized format.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

FITBIT_API_BASE_URL = "https://api.fitbit.com"
FITBIT_AUTHORIZE_URL = "https://www.fitbit.com/oauth2/authorize"
FITBIT_TOKEN_URL = "https://api.fitbit.com/oauth2/token"

logger = logging.getLogger(__name__)


class FitbitClientError(RuntimeError):
    """Raised when Fitbit API requests fail or responses cannot be parsed."""


@dataclass
class FitbitCredentials:
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: str = "activity heartrate sleep"


@dataclass
class FitbitToken:
    access_token: str
    refresh_token: str
    expires_at: datetime
    scope: str
    user_id: str

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at

    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "FitbitToken":
        expires_in = response.get("expires_in", 0)
        return cls(
            access_token=response["access_token"],
            refresh_token=response["refresh_token"],
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
            scope=response.get("scope", ""),
            user_id=response.get("user_id", ""),
        )


class FitbitClient:
    def __init__(
        self,
        credentials: FitbitCredentials,
        token: Optional[FitbitToken] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.credentials = credentials
        self.token = token
        self.session = session or requests.Session()

    # OAuth helpers -----------------------------------------------------
    def get_authorize_url(self, state: Optional[str] = None) -> str:
        params = {
            "client_id": self.credentials.client_id,
            "response_type": "code",
            "scope": self.credentials.scope,
            "redirect_uri": self.credentials.redirect_uri,
            "expires_in": "604800",
        }

        if state:
            params["state"] = state

        request = requests.Request("GET", FITBIT_AUTHORIZE_URL, params=params).prepare()
        return request.url

    def exchange_code_for_token(self, authorization_code: str) -> FitbitToken:
        headers = {
            "Authorization": f"Basic {self._encoded_credentials()}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "client_id": self.credentials.client_id,
            "grant_type": "authorization_code",
            "redirect_uri": self.credentials.redirect_uri,
            "code": authorization_code,
        }

        response = self.session.post(FITBIT_TOKEN_URL, headers=headers, data=data)
        self._raise_for_status(response)

        token_payload = response.json()
        self.token = FitbitToken.from_response(token_payload)
        return self.token

    def refresh_access_token(self) -> FitbitToken:
        if not self.token:
            raise FitbitClientError("Cannot refresh token before initial authentication.")

        headers = {
            "Authorization": f"Basic {self._encoded_credentials()}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.token.refresh_token,
        }

        response = self.session.post(FITBIT_TOKEN_URL, headers=headers, data=data)
        self._raise_for_status(response)

        token_payload = response.json()
        self.token = FitbitToken.from_response(token_payload)
        return self.token

    # Data retrieval ----------------------------------------------------
    def get_heart_rate_intraday(self, date: datetime, detail_level: str = "1min") -> List[Dict[str, Any]]:
        endpoint = (
            f"/1/user/-/activities/heart/date/{date.strftime('%Y-%m-%d')}/1d/{detail_level}.json"
        )
        payload = self._get(endpoint)
        dataset = payload.get("activities-heart-intraday", {}).get("dataset", [])
        for item in dataset:
            item["datetime"] = datetime.combine(date.date(), datetime.strptime(item["time"], "%H:%M:%S").time())
        return dataset

    def get_daily_activity_summary(self, date: datetime) -> Dict[str, Any]:
        endpoint = f"/1/user/-/activities/date/{date.strftime('%Y-%m-%d')}.json"
        return self._get(endpoint)

    def get_sleep_logs(self, date: datetime) -> List[Dict[str, Any]]:
        endpoint = f"/1.2/user/-/sleep/date/{date.strftime('%Y-%m-%d')}.json"
        payload = self._get(endpoint)
        return payload.get("sleep", [])

    # Utility helpers ---------------------------------------------------
    def _authorized_headers(self) -> Dict[str, str]:
        if not self.token:
            raise FitbitClientError("No access token available. Authenticate first.")
        if self.token.is_expired:
            logger.info("Refreshing expired Fitbit access token.")
            self.refresh_access_token()

        return {"Authorization": f"Bearer {self.token.access_token}"}

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{FITBIT_API_BASE_URL}{endpoint}"
        headers = self._authorized_headers()
        response = self.session.get(url, headers=headers, params=params)
        self._raise_for_status(response)
        return response.json()

    def _encoded_credentials(self) -> str:
        creds = f"{self.credentials.client_id}:{self.credentials.client_secret}"
        return base64.b64encode(creds.encode()).decode()

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            message = ""
            try:
                message = response.json()
            except Exception:  # noqa: BLE001
                message = response.text
            logger.error("Fitbit API error: %s", message)
            raise FitbitClientError(f"Fitbit API error: {message}") from error


