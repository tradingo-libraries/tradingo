"""Utility functions for sending emails using SMTP."""

import smtplib
from contextlib import contextmanager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Generator

from tradingo.settings import SMTPConfig


@contextmanager
def smtp_connection() -> Generator[smtplib.SMTP, Any, Any]:
    """Context manager to create and yield an SMTP connection."""
    config = SMTPConfig.from_env()
    server = smtplib.SMTP(config.server_uri, config.port)
    try:
        server.starttls()
        server.login(config.username, config.password)
        yield server
    finally:
        server.quit()


def send_email(
    body: str,
    subject: str,
    recipient: str,
) -> None:
    """
    Send an email with the given body to recipient.

    Args:
        body (str): HTML content to be sent in the email.
        subject (str): Subject of the email.
        recipient (str): Email address of the recipient.
    """

    config = SMTPConfig.from_env()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.username
    msg["To"] = recipient
    msg.attach(MIMEText(body, "html"))

    with smtp_connection() as server:
        server.sendmail(config.username, recipient, msg.as_string())
