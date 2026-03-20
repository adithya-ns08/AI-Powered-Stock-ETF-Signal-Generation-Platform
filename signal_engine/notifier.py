"""
signal_engine/notifier.py
==========================
Real-Time Alerting System for AI Stock Signal Platform.
"""

import os
import json
import logging
import requests
from pathlib import Path

class SlackNotifier:
    def __init__(self, webhook_url: str = None, state_file: str = "signal_state.json"):
        """
        Initialize the Slack Notifier.
        :param webhook_url: Slack incoming webhook URL. Falls back to STREAMLIT SECRETS.
        :param state_file: Local JSON file to persist the previous signal states.
        """
        self.state_file = Path(state_file)
        self.logger = logging.getLogger(__name__)

        # Securely parse webhook URL from Streamlit Secrets manager
        if webhook_url:
            self.webhook_url = webhook_url
        else:
            try:
                import streamlit as st
                self.webhook_url = st.secrets.get("SLACK_WEBHOOK_URL")
            except Exception as e:
                self.webhook_url = None
                self.logger.warning(f"Could not load webhook URL from st.secrets: {e}")

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load state file: {e}")
        return {}

    def _save_state(self, state: dict):
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save state file: {e}")

    def send_alert(self, ticker: str, signal: str, price: float, confidence: float):
        """
        Send a formatted POST request to the Slack Webhook using Attachments.
        """
        if not self.webhook_url:
            self.logger.warning("Slack webhook URL not configured. Skipping alert.")
            return

        # Explicitly code the left border color
        color = "#00e676" if signal.lower() == "buy" else "#ff1744"
        
        # Slack attachments structure
        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"🚨 *AI Signal Alert*: {ticker.upper()} has shifted to *{signal.upper()}* at ${price:.2f}. Model Confidence: {confidence}%"
                            }
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            self.logger.info(f"Successfully sent Slack alert for {ticker}.")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send Slack alert for {ticker}: {e}")

    def process_new_signals(self, latest_signals: list):
        """
        Compare new incoming Streamlit SIGNALS dict against the previous state.
        Triggers send_alert() only if the signal logic has changed. Anti-Spam protection.
        """
        previous_state = self._load_state()
        new_state = {}
        changes_detected = False

        for sig_data in latest_signals:
            ticker = sig_data.get("ticker", "UNKNOWN")
            current_signal = sig_data.get("signal", "hold").lower()
            price = sig_data.get("price", 0.0)
            confidence = sig_data.get("confidence", 0.0)

            new_state[ticker] = current_signal
            old_signal = previous_state.get(ticker, "hold").lower()

            # Anti-Spam: Only send if signal changed from Hold -> Buy/Sell, or Buy -> Sell, etc.
            # Usually we only want to alert on Buy/Sell, ignoring Hold alerts unless desired.
            if current_signal != old_signal and current_signal != "hold":
                self.send_alert(ticker, current_signal, price, confidence)
                changes_detected = True

        # Always update memory for next run
        if changes_detected or not previous_state:
            # We also save if there wasn't a previous state (initialization)
            # So next tick knows we're already holding the current states.
            self._save_state(new_state)
