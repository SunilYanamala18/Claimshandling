"""
Email monitoring + Azure DI extraction + Simple JSON storage + FastAPI endpoints.

Usage:
  1. Create a .env with EMAIL_ADDRESS, EMAIL_PASSWORD, AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_API_KEY
  2. python -m pip install fastapi uvicorn python-multipart python-dotenv azure-ai-formrecognizer azure-core
  3. uvicorn app:app --reload
"""

import imaplib
import email
import time
import os
import json
import uuid
import logging
import tempfile
import threading
from typing import List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# ---------- Config & Logging ----------
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")

DATA_DIR = Path("data")
EMAIL_JSON_DIR = DATA_DIR / "emails"
ATTACHMENTS_DIR = DATA_DIR / "attachments"
EMAIL_JSON_DIR.mkdir(parents=True, exist_ok=True)
ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("email-processor")

# simple Pydantic models for API responses
class AttachmentInfo(BaseModel):
    filename: str
    file_type: str
    extracted_data: dict | None = None
    stored_path: str

class StoredEmail(BaseModel):
    id: str
    from_addr: str | None
    to_addr: str | None
    subject: str | None
    date: str | None
    body: str | None
    attachments: List[AttachmentInfo]
    stored_at: float

# ---------- Email Processor (writes JSON + attachments) ----------
class EmailProcessor:
    def __init__(self, email_address: str, password: str, azure_endpoint: str, azure_key: str):
        self.email_address = email_address
        self.password = password
        self.azure_client = DocumentAnalysisClient(endpoint=azure_endpoint, credential=AzureKeyCredential(azure_key))
        self._connect_imap()

    def _connect_imap(self):
        logger.info("Connecting to Gmail IMAP...")
        self.mail = imaplib.IMAP4_SSL("imap.gmail.com")
        self.mail.login(self.email_address, self.password)
        self.mail.select("inbox")
        logger.info("IMAP connected")

    def check_new_emails(self):
        try:
            status, messages = self.mail.search(None, "UNSEEN")
            email_ids = messages[0].split() if messages and messages[0] else []
            logger.info("Found %d new emails", len(email_ids))
            return email_ids
        except Exception as e:
            logger.error("Error searching emails: %s", e)
            return []

    def read_email(self, email_id) -> dict:
        """Return a processed email dict (and save attachments + JSON)."""
        status, msg_data = self.mail.fetch(email_id, "(RFC822)")
        if not msg_data or not msg_data[0]:
            raise RuntimeError("Failed to fetch message data")

        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        email_dict = {
            "id": str(uuid.uuid4()),
            "from": msg.get("From"),
            "to": msg.get("To"),
            "subject": msg.get("Subject"),
            "date": msg.get("Date"),
            "body": self._get_body(msg),
            "attachments": [],
            "stored_at": time.time(),
        }

        # process attachments
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get("Content-Disposition") is None:
                continue

            filename = part.get_filename()
            if not filename:
                continue

            # ensure unique filename on disk
            ts = int(time.time())
            safe_filename = f"{ts}_{uuid.uuid4().hex}_{filename}"
            stored_path = str((ATTACHMENTS_DIR / safe_filename).resolve())

            with open(stored_path, "wb") as f:
                f.write(part.get_payload(decode=True))

            # detect type
            ext = filename.lower()
            if ext.endswith(".pdf"):
                file_type = "PDF"
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                file_type = "IMAGE"
            else:
                file_type = "UNSUPPORTED"

            extracted = None
            if file_type in ("PDF", "IMAGE"):
                try:
                    extracted = self._extract_with_azure(stored_path)
                except Exception as e:
                    logger.exception("Azure extraction failed for %s: %s", filename, e)
                    extracted = {"error": str(e)}

            attachment_info = {
                "filename": filename,
                "file_type": file_type,
                "extracted_data": extracted,
                "stored_path": stored_path,
            }

            email_dict["attachments"].append(attachment_info)

        # save JSON
        json_path = EMAIL_JSON_DIR / f"{email_dict['id']}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(email_dict, jf, ensure_ascii=False, indent=2)

        logger.info("Stored email: %s -> %s (attachments=%d)", email_dict["subject"], json_path, len(email_dict["attachments"]))

        return email_dict

    def _get_body(self, msg) -> str:
        if msg.is_multipart():
            # prefer text/plain if available
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and not part.get_filename():
                    try:
                        return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                    except Exception:
                        return part.get_payload(decode=True).decode(errors="ignore")
            # fallback html
            for part in msg.walk():
                if part.get_content_type() == "text/html" and not part.get_filename():
                    try:
                        return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                    except Exception:
                        return part.get_payload(decode=True).decode(errors="ignore")
            return ""
        else:
            try:
                return msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="ignore")
            except Exception:
                return msg.get_payload(decode=True).decode(errors="ignore")

    def _extract_with_azure(self, filepath: str) -> dict:
        """Use Azure prebuilt-read model for OCR; returns dict {content: ...}"""
        logger.info("Azure extract: %s", filepath)
        with open(filepath, "rb") as f:
            poller = self.azure_client.begin_analyze_document("prebuilt-read", document=f)
            result = poller.result()
        return {"content": result.content} if result else {"content": ""}

    def monitor_inbox(self, interval: int = 10):
        logger.info("Starting monitor loop (interval=%s)", interval)
        while True:
            try:
                # keep alive / reconnect if needed
                try:
                    self.mail.noop()
                except Exception as e:
                    logger.warning("IMAP noop failed (%s) - reconnecting", e)
                    self._connect_imap()

                # ensure inbox selected
                self.mail.select("inbox")
                email_ids = self.check_new_emails()
                for eid in email_ids:
                    try:
                        logger.info("Processing email id: %s", eid)
                        email_dict = self.read_email(eid)
                        # mark seen
                        self.mail.store(eid, "+FLAGS", "\\Seen")
                    except Exception as e:
                        logger.exception("Failed to process email %s: %s", eid, e)

                time.sleep(interval)
            except Exception as e:
                logger.exception("Monitor loop error: %s", e)
                time.sleep(5)

# ---------- FastAPI app ----------
app = FastAPI(title="Email Extraction (JSON store)")

# Create global processor instance lazily later (when env present)
processor: EmailProcessor | None = None

@app.on_event("startup")
def startup_event():
    global processor
    # validate env
    missing = [name for name, val in [
        ("EMAIL_ADDRESS", EMAIL_ADDRESS),
        ("EMAIL_PASSWORD", EMAIL_PASSWORD),
        ("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", AZURE_ENDPOINT),
        ("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", AZURE_KEY),
    ] if not val]
    if missing:
        logger.critical("Missing env vars: %s", missing)
        raise RuntimeError(f"Missing env vars: {missing}")

    processor = EmailProcessor(EMAIL_ADDRESS, EMAIL_PASSWORD, AZURE_ENDPOINT, AZURE_KEY)
    # start monitor thread (daemon so it won't block process exit)
    t = threading.Thread(target=processor.monitor_inbox, kwargs={"interval": 10}, daemon=True)
    t.start()
    logger.info("Background email monitor thread started")

@app.get("/emails", response_model=List[StoredEmail])
def list_emails():
    """List processed emails (reads JSON files)."""
    files = sorted(EMAIL_JSON_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    result = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                # map to StoredEmail model fields
                attachments = [
                    AttachmentInfo(
                        filename=a["filename"],
                        file_type=a["file_type"],
                        extracted_data=a.get("extracted_data"),
                        stored_path=a.get("stored_path"),
                    )
                    for a in data.get("attachments", [])
                ]
                stored = StoredEmail(
                    id=data["id"],
                    from_addr=data.get("from"),
                    to_addr=data.get("to"),
                    subject=data.get("subject"),
                    date=data.get("date"),
                    body=data.get("body"),
                    attachments=attachments,
                    stored_at=data.get("stored_at", 0),
                )
                result.append(stored)
        except Exception as e:
            logger.warning("Failed to load email json %s: %s", f, e)
    return result

@app.get("/emails/{email_id}", response_model=StoredEmail)
def get_email(email_id: str):
    p = EMAIL_JSON_DIR / f"{email_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Email not found")
    with open(p, "r", encoding="utf-8") as jf:
        data = json.load(jf)
    attachments = [
        AttachmentInfo(
            filename=a["filename"],
            file_type=a["file_type"],
            extracted_data=a.get("extracted_data"),
            stored_path=a.get("stored_path"),
        )
        for a in data.get("attachments", [])
    ]
    return StoredEmail(
        id=data["id"],
        from_addr=data.get("from"),
        to_addr=data.get("to"),
        subject=data.get("subject"),
        date=data.get("date"),
        body=data.get("body"),
        attachments=attachments,
        stored_at=data.get("stored_at", 0),
    )

@app.get("/emails/{email_id}/attachments/{filename}")
def download_attachment(email_id: str, filename: str):
    """Download an attachment file previously stored for an email."""
    p = EMAIL_JSON_DIR / f"{email_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Email not found")

    with open(p, "r", encoding="utf-8") as jf:
        data = json.load(jf)

    # find matching attachment by original filename
    for a in data.get("attachments", []):
        if a.get("filename") == filename:
            stored_path = a.get("stored_path")
            if stored_path and os.path.exists(stored_path):
                return FileResponse(stored_path, media_type="application/octet-stream", filename=filename)
            else:
                raise HTTPException(status_code=404, detail="Attachment file missing on server")

    raise HTTPException(status_code=404, detail="Attachment not found in email record")

@app.get("/attachments/{file_uuid_name}")
def download_attachment_by_name(file_uuid_name: str):
    """Directly download an attachment by stored filename (the stored file name includes timestamp+uuid)."""
    candidate = ATTACHMENTS_DIR / file_uuid_name
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="File not found")
    orig_name = "_".join(file_uuid_name.split("_")[2:]) if "_" in file_uuid_name else file_uuid_name
    return FileResponse(str(candidate.resolve()), media_type="application/octet-stream", filename=orig_name)
