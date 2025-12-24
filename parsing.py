"""
COMPLETE EMAIL READING SYSTEM
Monitors Gmail inbox and processes emails with Azure Document Intelligence
"""

import imaplib
import email
import time
import os
import logging
import tempfile
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


class EmailProcessor:
    def __init__(self, email_address, password, azure_endpoint, azure_key):
        self.email_address = email_address
        self.password = password

        # ---- Setup IMAP connection ----
        self._connect_imap()

        # ---- Setup Azure Document Intelligence ----
        self.doc_client = DocumentAnalysisClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_key)
        )

    def _connect_imap(self):
        """Connect to Gmail IMAP fresh"""
        logging.info("Connecting to Gmail IMAP...")
        try:
            self.mail = imaplib.IMAP4_SSL("imap.gmail.com")
            self.mail.login(self.email_address, self.password)
            self.mail.select("inbox")
            logging.info("IMAP connected successfully")
        except imaplib.IMAP4.error as e:
            logging.critical("IMAP authentication or protocol error: %s", e)
            raise
        except Exception as e:
            logging.critical("Unexpected IMAP connection error: %s", e)
            raise

    def check_new_emails(self):
        """Search for unread emails"""
        try:
            status, messages = self.mail.search(None, "UNSEEN")
            email_ids = messages[0].split()
            logging.info("Found %d new emails", len(email_ids))
            return email_ids

        except Exception as e:
            logging.error("Error searching emails: %s", e)
            return []

    def read_email(self, email_id):
        """Reads and extracts email content + attachments"""
        status, msg_data = self.mail.fetch(email_id, "(RFC822)")
        email_message = email.message_from_bytes(msg_data[0][1])

        email_data = {
            "from": email_message.get("From"),
            "to": email_message.get("To"),
            "subject": email_message.get("Subject"),
            "date": email_message.get("Date"),
            "body": self._get_body(email_message),
            "attachments": []
        }

        # -------- Process Attachments --------
        for part in email_message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get("Content-Disposition") is None:
                continue

            filename = part.get_filename()
            if not filename:
                continue

            temp_dir = tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)
            timestamp = int(time.time())
            filepath = os.path.join(temp_dir, f"{timestamp}_{filename}")

            with open(filepath, "wb") as f:
                f.write(part.get_payload(decode=True))

            # Detect attachment type
            ext = filename.lower()
            if ext.endswith(".pdf"):
                file_type = "PDF"
                logging.info("Attachment found: %s (PDF)", filename)

            elif ext.endswith((".png", ".jpg", ".jpeg")):
                file_type = "IMAGE"
                logging.info("Attachment found: %s (Image)", filename)

            else:
                file_type = "UNSUPPORTED"
                logging.warning("Attachment found: %s (Unsupported Type)", filename)

            attachment_data = {
                "filename": filename,
                "file_type": file_type
            }

            # Process only supported types
            if file_type in ["PDF", "IMAGE"]:
                attachment_data["extracted_data"] = self._extract_with_azure(filepath)

            email_data["attachments"].append(attachment_data)

        return email_data

    def _get_body(self, msg):
        """Extract plain or HTML body"""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode(errors="ignore")
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    return part.get_payload(decode=True).decode(errors="ignore")

        else:
            return msg.get_payload(decode=True).decode(errors="ignore")

        return ""

    def _extract_with_azure(self, filepath):
        """Extract text using Azure Read Model"""
        try:
            logging.info("Extracting content with Azure: %s", filepath)

            with open(filepath, "rb") as f:
                poller = self.doc_client.begin_analyze_document(
                    "prebuilt-read", document=f
                )
                result = poller.result()

            return {"content": result.content}

        except Exception as e:
            logging.error("Azure extraction error: %s", e)
            return {"error": str(e)}
        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as cleanup_error:
                logging.warning("Failed to remove temporary file %s: %s", filepath, cleanup_error)

    def monitor_inbox(self, interval=10):
        """Continuously monitor Gmail inbox for new emails"""
        logging.info("Monitoring inbox for: %s", self.email_address)
        logging.info("Checking every %s seconds", interval)

        while True:
            try:
                # ---- IMAP Keep Alive ----
                try:
                    self.mail.noop()
                except Exception as noop_error:
                    logging.warning("IMAP disconnected (%s) → reconnecting...", noop_error)
                    self._connect_imap()

                # Select inbox every cycle
                self.mail.select("inbox")

                # Check for unread emails
                email_ids = self.check_new_emails()

                for email_id in email_ids:
                    logging.info("Reading email ID: %s", email_id)

                    email_data = self.read_email(email_id)

                    logging.info("From: %s", email_data["from"])
                    logging.info("Subject: %s", email_data["subject"])
                    logging.info("Body preview: %s...", email_data["body"][:200])

                    # Mark as read
                    self.mail.store(email_id, "+FLAGS", "\\Seen")

                time.sleep(interval)

            except Exception as e:
                logging.exception("Loop error: %s", e)
                time.sleep(5)


def _load_config_from_env():
    email_address = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_PASSWORD")
    azure_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    azure_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")

    missing = [
        name
        for name, value in [
            ("EMAIL_ADDRESS", email_address),
            ("EMAIL_PASSWORD", password),
            ("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", azure_endpoint),
            ("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", azure_key),
        ]
        if not value
    ]

    if missing:
        logging.critical("Missing required environment variables: %s", ", ".join(missing))
        raise SystemExit(1)

    return email_address, password, azure_endpoint, azure_key


def main():
    email_address, password, azure_endpoint, azure_key = _load_config_from_env()

    try:
        processor = EmailProcessor(
            email_address=email_address,
            password=password,
            azure_endpoint=azure_endpoint,
            azure_key=azure_key,
        )
        processor.monitor_inbox(interval=10)
    except KeyboardInterrupt:
        logging.info("Shutting down email processor due to keyboard interrupt")
    except SystemExit:
        raise
    except Exception as e:
        logging.exception("Fatal error in email processor: %s", e)
        raise


if __name__ == "__main__":
    main()
