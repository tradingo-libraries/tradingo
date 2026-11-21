"""Utility functions for sending emails using SMTP."""

import mimetypes
import smtplib
from contextlib import contextmanager
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Generator, Sequence, Tuple, Union

from tradingo.settings import SMTPConfig

# Either a path to a file on disk, or a (filename, content) pair for content
# built in memory (e.g. redacted text).
Attachment = Union[str, Path, Tuple[str, bytes]]
# (source, content_id) — content_id is referenced from the HTML body as "cid:<content_id>".
InlineImage = Tuple[Attachment, str]


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


def _read_attachment(item: Attachment) -> tuple[str, bytes] | None:
    if isinstance(item, tuple):
        return item
    path = Path(item)
    if not path.is_file():
        return None
    return path.name, path.read_bytes()


def _mime_part(
    filename: str, data: bytes, *, disposition: str, content_id: str | None = None
) -> MIMEBase:
    ctype, _ = mimetypes.guess_type(filename)
    maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
    part: MIMEBase
    if maintype == "image":
        part = MIMEImage(data, _subtype=subtype or "octet-stream")
    else:
        part = MIMEApplication(data, _subtype=subtype or "octet-stream")
    if content_id:
        part.add_header("Content-ID", f"<{content_id}>")
    part.add_header("Content-Disposition", disposition, filename=filename)
    return part


def send_email(
    body: str,
    subject: str,
    recipient: str,
    attachments: Sequence[Attachment] | None = None,
    inline_images: Sequence[InlineImage] | None = None,
) -> None:
    """
    Send an email with the given body to recipient.

    Args:
        body (str): HTML content to be sent in the email. May reference inline
            images via ``<img src="cid:...">`` — see ``inline_images``.
        subject (str): Subject of the email.
        recipient (str): Email address of the recipient.
        attachments (Sequence[Attachment] | None): Files to attach — either
            filesystem paths, or (filename, bytes) pairs for in-memory content.
            Paths that don't exist are silently skipped.
        inline_images (Sequence[InlineImage] | None): Images to embed inline,
            referenced from ``body`` by ``cid:<content_id>``.
    """

    config = SMTPConfig.from_env()

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = config.username
    msg["To"] = recipient

    related = MIMEMultipart("related")
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(body, "html"))
    related.attach(alt)
    for source, content_id in inline_images or []:
        read = _read_attachment(source)
        if read is None:
            continue
        filename, data = read
        related.attach(
            _mime_part(filename, data, disposition="inline", content_id=content_id)
        )
    msg.attach(related)

    for item in attachments or []:
        read = _read_attachment(item)
        if read is None:
            continue
        filename, data = read
        msg.attach(_mime_part(filename, data, disposition="attachment"))

    with smtp_connection() as server:
        server.sendmail(config.username, recipient, msg.as_string())
