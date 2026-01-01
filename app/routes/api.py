
import json
from datetime import datetime, timezone
from flask import Blueprint, jsonify, session, current_app
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from ..utils.email_utils import sanitize_for_json
from ..services.gmail_service import process_unread_emails

api_bp = Blueprint("api", __name__)

@api_bp.route("/unread", methods=["GET"])
def unread():
    creds_json = session.get("credentials")
    if not creds_json:
        return jsonify({"ok": False, "error": "Not authenticated. Please login."}), 401

    try:
        info = json.loads(creds_json)
        creds = Credentials.from_authorized_user_info(info, current_app.config["SCOPES"])

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            session["credentials"] = creds.to_json()

        service = build("gmail", "v1", credentials=creds)

        ml_service = current_app.extensions.get("ml_service")
        sn_service = current_app.extensions.get("sn_service")

        result = process_unread_emails(service, ml_service, sn_service, current_app.logger)
        return jsonify({"ok": True, "result": sanitize_for_json(result)})

    except Exception as e:
        current_app.logger.exception("Unread processing failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@api_bp.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})
