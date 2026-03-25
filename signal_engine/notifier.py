"""
signal_engine/notifier.py
==========================
Real-Time Alerting System for AI Stock Signal Platform.
"""

import os
import json
import logging
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path

class AlertManager:
    def __init__(self, webhook_url: str | None = None, email_address: str | None = None, threshold_pct: float | None = None, state_file: str = "signal_state.json"):
        """
        Initialize the Alert Manager.
        :param webhook_url: Slack incoming webhook URL.
        :param email_address: Target email address for notifications.
        :param threshold_pct: Float percentage to trigger threshold alert.
        :param state_file: Local JSON file to persist the previous signal states.
        """
        self.state_file = Path(state_file)
        self.logger = logging.getLogger(__name__)

        self.webhook_url = webhook_url
        self.email_address = email_address
        self.threshold_pct = threshold_pct

        # If webhook not provided, try secrets
        if not self.webhook_url:
            try:
                import streamlit as st
                self.webhook_url = st.secrets.get("SLACK_WEBHOOK_URL")
            except Exception:
                pass

        # Email SMTP Setup from secrets if available
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.smtp_user = None
        self.smtp_pass = None
        try:
            import streamlit as st
            self.smtp_user = st.secrets.get("SMTP_USER", "")
            self.smtp_pass = st.secrets.get("SMTP_PASSWORD", "")
            self.smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
            self.smtp_port = st.secrets.get("SMTP_PORT", 587)
        except Exception:
            pass

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

    def _is_rate_limited(self, ticker: str, alert_type: str, state: dict) -> bool:
        """
        Check if an alert for this ticker+type was sent within the last hour.
        """
        key = f"{ticker}_{alert_type}_timestamp"
        last_time_str = state.get(key)
        if last_time_str:
            try:
                last_time = datetime.fromisoformat(last_time_str)
                if datetime.now() - last_time < timedelta(hours=1):
                    return True
            except Exception:
                pass
        return False

    def _update_rate_limit(self, ticker: str, alert_type: str, state: dict):
        """Update the timestamp in state."""
        key = f"{ticker}_{alert_type}_timestamp"
        state[key] = datetime.now().isoformat()

    def send_external_alert(self, ticker: str, signal: str, price: float, extra_msg: str = ""):
        """
        Send a notification via Slack and/or Email.
        signal: 'buy', 'sell', or 'threshold'
        """
        # Determine color
        color = "#00e676" if signal.lower() == "buy" else "#ff1744"
        if signal.lower() == "threshold":
            color = "#ffab00"

        title = f"🚀 NEW SIGNAL: {signal.upper()} {ticker} at ${price:.2f}"
        if signal.lower() == "threshold":
            title = f"⚠️ PRICE ALERT: {ticker} moved significantly. Current: ${price:.2f}"
            
        full_msg = ""
        if extra_msg:
            full_msg += f"{extra_msg}"

        # 1. Slack
        if self.webhook_url:
            self._send_slack(title, full_msg, color)

        # 2. Email
        if self.email_address:
            self._send_email(title, full_msg)

    def _send_slack(self, title: str, text: str, color: str):
        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*{title}*\n{text}"
                            }
                        }
                    ]
                }
            ]
        }
        try:
            response = requests.post(self.webhook_url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
            response.raise_for_status()
            self.logger.info(f"Successfully sent Slack alert: {title}")
        except Exception as e:
            self.logger.error(f"Slack alert failed: {e}")

    def _send_email(self, subject: str, body: str):
        # We need self.smtp_user and pass to actually send.
        if not self.smtp_user or not self.smtp_pass:
            self.logger.warning("SMTP credentials not configured. Skipping actual email dispatch.")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.email_address
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            self.logger.info(f"Successfully sent Email alert: {subject}")
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")

    def process_new_signals(self, latest_signals: list):
        """
        Process the incoming Streamlit SIGNALS dict against the previous state.
        Triggers send_alert() only if the signal logic has changed. Anti-Spam protection.
        """
        state = self._load_state()
        changes_detected = False

        for sig_data in latest_signals:
            ticker = sig_data.get("ticker", "UNKNOWN")
            current_signal = sig_data.get("signal", "hold").lower()
            price = sig_data.get("price", 0.0)
            conf = sig_data.get("confidence", 0.0)
            
            old_signal = state.get(f"{ticker}_signal", "hold").lower()
            state[f"{ticker}_signal"] = current_signal

            if current_signal != old_signal and current_signal in ["buy", "sell"]:
                # Check rate limit
                if not self._is_rate_limited(ticker, current_signal, state):
                    msg = f"Model Confidence: {conf}%"
                    self.send_external_alert(ticker, current_signal, price, msg)
                    self._update_rate_limit(ticker, current_signal, state)
                changes_detected = True

        if changes_detected or not state:
            self._save_state(state)

    def process_price_alert(self, ticker: str, current_price: float, pct_change: float):
        """
        Threshold-Based Price Alerts. Trigger if abs(pct_change) >= threshold_pct.
        """
        if not self.threshold_pct:
            return

        if abs(pct_change) >= self.threshold_pct:
            state = self._load_state()
            if not self._is_rate_limited(ticker, "threshold", state):
                msg = f"Price moved by {pct_change:+.2f}% compared to previous close. Threshold is {self.threshold_pct}%."
                self.send_external_alert(ticker, "threshold", current_price, msg)
                self._update_rate_limit(ticker, "threshold", state)
                self._save_state(state)

    def test_notification(self):
        """Triggered by the UI 'Send Test Alert' button."""
        self.send_external_alert("TEST_TICKER", "buy", 253.15, "This is a TEST notification from the Alert Settings panel.")
