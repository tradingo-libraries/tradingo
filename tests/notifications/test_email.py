"""Tests for tradingo.notifications.email"""

import smtplib
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from tradingo.notifications.email import send_email, smtp_connection
from tradingo.settings import SMTPConfig

MOCK_CONFIG = SMTPConfig(
    server_uri="smtp.example.com",
    port=587,
    username="sender@example.com",
    password="secret",
    app_prefix="smtp_",
)


@pytest.fixture
def mock_smtp_config() -> Generator[None, None, None]:
    with patch(
        "tradingo.notifications.email.SMTPConfig.from_env", return_value=MOCK_CONFIG
    ):
        yield


@pytest.fixture
def mock_smtp_server() -> Generator[MagicMock, None, None]:
    server = MagicMock(spec=smtplib.SMTP)
    with patch("tradingo.notifications.email.smtplib.SMTP", return_value=server):
        yield server


def test_smtp_connection_lifecycle(
    mock_smtp_config: None, mock_smtp_server: MagicMock
) -> None:
    """Connection is opened, authenticated, and closed cleanly."""
    with smtp_connection() as server:
        assert server is mock_smtp_server

    mock_smtp_server.starttls.assert_called_once()
    mock_smtp_server.login.assert_called_once_with(
        MOCK_CONFIG.username, MOCK_CONFIG.password
    )
    mock_smtp_server.quit.assert_called_once()


def test_smtp_connection_quits_on_exception(
    mock_smtp_config: None, mock_smtp_server: MagicMock
) -> None:
    """quit() is called even if an exception is raised inside the context."""
    with pytest.raises(RuntimeError):
        with smtp_connection():
            raise RuntimeError("boom")

    mock_smtp_server.quit.assert_called_once()


def test_send_email_builds_correct_message(
    mock_smtp_config: None, mock_smtp_server: MagicMock
) -> None:
    """send_email sends to the right recipient with correct subject and from."""
    send_email(
        body="<p>hello</p>",
        subject="Test Subject",
        recipient="recipient@example.com",
    )

    mock_smtp_server.sendmail.assert_called_once()
    from_addr, to_addr, raw_msg = mock_smtp_server.sendmail.call_args[0]
    assert from_addr == MOCK_CONFIG.username
    assert to_addr == "recipient@example.com"
    assert "Test Subject" in raw_msg
    assert "hello" in raw_msg


def test_send_email_uses_html_mime_type(
    mock_smtp_config: None, mock_smtp_server: MagicMock
) -> None:
    """Body is attached as text/html."""
    send_email(body="<b>bold</b>", subject="s", recipient="r@example.com")

    _, _, raw_msg = mock_smtp_server.sendmail.call_args[0]
    assert "text/html" in raw_msg
