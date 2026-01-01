
import os

class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "CHANGE_THIS_SECRET_KEY")

    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.send",
    ]

    CLIENT_SECRETS_FILE = os.getenv("GOOGLE_CLIENT_SECRETS", "credentials.json")
    OAUTHLIB_INSECURE_TRANSPORT = os.getenv("OAUTHLIB_INSECURE_TRANSPORT", "1")

    MODEL_DIR = os.getenv("MODEL_DIR", "model/model_artifacts")

    # ServiceNow
    SN_INSTANCE = os.getenv("SN_INSTANCE")        # e.g., dev202768
    SN_USER = os.getenv("SN_USER")                # e.g., admin
    SN_PASSWORD = os.getenv("SN_PASSWORD")        # e.g., your PDI password

    # Fields to request from SN Incident API
    SN_FIELDS = "number,short_description,sys_id,priority,opened_at"

