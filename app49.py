import os
import re
import json
import base64
import pickle
import unicodedata
import numpy as np
from typing import Optional
from email.mime.text import MIMEText
from datetime import datetime, timezone
from flask import Flask, redirect, url_for, session, jsonify, request, render_template_string
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

import requests
import faiss
from sentence_transformers import SentenceTransformer

# Optional: load env from .env (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- CONFIG ---------------- #
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "CHANGE_THIS_SECRET_KEY")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.labels",
]

CLIENT_SECRETS_FILE = os.getenv("GOOGLE_CLIENT_SECRETS", "credentials.json")
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = os.getenv("OAUTHLIB_INSECURE_TRANSPORT", "1")

# ---------------- BE SAFE PASSWORD RESET CONFIG ---------------- #
BESAFE_KEYWORDS = [
    "be safe",
    "besafe",
    "be-safe",
    "password reset",
    "password",
    "reset password",
    "reset",
    "authentication",
    "mfa",
    "pingid",
    "ping id",
]

# ---------------- LOAD ML ---------------- #
MODEL_DIR = os.getenv("MODEL_DIR", "model/model_artifacts")
faiss_index = None
df = None
model = None

try:
    faiss_index = faiss.read_index(f"{MODEL_DIR}/email_support_index.faiss")
    df = pickle.load(open(f"{MODEL_DIR}/email_support_data.pkl", "rb"))
    model = SentenceTransformer(f"{MODEL_DIR}/embedding_model")
    app.logger.info("ML artifacts loaded.")
except Exception as e:
    app.logger.warning(f"ML artifacts not loaded. Falling back to incident creation. Reason: {e}")

# ---------------- UTILITIES ---------------- #



# ---------------- DOMAIN BLOCKLIST CONFIG ---------------- #
BLOCKED_DOMAINS = [
    d.strip().lower()
    for d in (os.getenv("BLOCKED_DOMAINS", "") or "").split(",")
    if d.strip()
]
BLOCKED_MARK_READ = os.getenv("BLOCKED_MARK_READ", "false").strip().lower() in ("1", "true", "yes")

NONBA_DOMAINS = [
    d.strip().lower()
    for d in (os.getenv("NONBA_DOMAINS", "false").split(","))
    if d.strip()
]

CC_SKIP_ADDRESSES = [
    addr.strip().lower()
    for addr in (os.getenv("CC_SKIP_ADDRESSES", "").split(","))
    if addr.strip()
]
# ────────────────────────────────────────
print("DEBUG ──────────────── CC_SKIP_ADDRESSES (raw env)   :", os.getenv("CC_SKIP_ADDRESSES", "NOT_SET"))
print("DEBUG ──────────────── CC_SKIP_ADDRESSES (parsed)   :", CC_SKIP_ADDRESSES)
print("DEBUG ──────────────── Length               :", len(CC_SKIP_ADDRESSES))
if CC_SKIP_ADDRESSES:
    print("First few items:", CC_SKIP_ADDRESSES[:3])
# ────────────────────────────────────────
# Debug: Log loaded CC skip addresses
if CC_SKIP_ADDRESSES:
    app.logger.info(f"CC_SKIP_ADDRESSES loaded: {CC_SKIP_ADDRESSES}")
else:
    app.logger.info("No CC_SKIP_ADDRESSES configured")
    
# NEW: Track closed incidents awaiting new incident creation
CLOSED_INCIDENT_PENDING_FILE = os.getenv("CLOSED_INCIDENT_PENDING_FILE", "closed_incident_pending.pkl")
closed_incident_pending = {}  # thread_id -> {"old_incident_number": ..., "sender": ..., "subject": ..., "timestamp": ...}

def _load_closed_incident_pending():
    global closed_incident_pending
    try:
        with open(CLOSED_INCIDENT_PENDING_FILE, "rb") as f:
            closed_incident_pending = pickle.load(f)
            app.logger.info(f"Loaded closed incident pending: {len(closed_incident_pending)} entries.")
    except Exception:
        closed_incident_pending = {}

def _save_closed_incident_pending():
    try:
        with open(CLOSED_INCIDENT_PENDING_FILE, "wb") as f:
            pickle.dump(closed_incident_pending, f)
    except Exception as e:
        app.logger.warning(f"Failed to persist closed incident pending: {e}")

def _mark_closed_incident_pending(thread_id: str, old_incident_number: str, sender: str, subject: str):
    """Mark a thread as waiting for new incident after closed incident."""
    closed_incident_pending[thread_id] = {
        "old_incident_number": old_incident_number,
        "sender": sender,
        "subject": subject,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    _save_closed_incident_pending()

def _is_closed_incident_pending(thread_id: str) -> bool:
    """Check if a thread is waiting for new incident creation after closed incident."""
    return thread_id in closed_incident_pending

def _get_closed_incident_info(thread_id: str) -> Optional[dict]:
    """Get closed incident info if pending."""
    return closed_incident_pending.get(thread_id)

def _clear_closed_incident_pending(thread_id: str):
    """Clear closed incident pending status after new incident creation."""
    if thread_id in closed_incident_pending:
        del closed_incident_pending[thread_id]
        _save_closed_incident_pending()

# Load the state when app starts
_load_closed_incident_pending()

# Add these new functions after the existing utility functions (around line 200)

def extract_attachments(service, msg_id: str, payload: dict) -> list:
    """
    Extract attachments from Gmail message.
    Returns list of dicts with 'filename', 'mimeType', and 'data' (base64).
    """
    attachments = []
    
    def process_parts(parts):
        for part in parts:
            # Check if this part has subparts
            if 'parts' in part:
                process_parts(part['parts'])
                continue
            
            # Check if this is an attachment
            filename = part.get('filename', '')
            mime_type = part.get('mimeType', '')
            
            if filename:  # Has a filename, likely an attachment
                body = part.get('body', {})
                attachment_id = body.get('attachmentId')
                
                if attachment_id:
                    try:
                        # Download the attachment
                        attachment = service.users().messages().attachments().get(
                            userId='me',
                            messageId=msg_id,
                            id=attachment_id
                        ).execute()
                        
                        file_data = attachment.get('data', '')
                        
                        attachments.append({
                            'filename': filename,
                            'mimeType': mime_type,
                            'data': file_data,  # Base64 encoded
                            'size': body.get('size', 0)
                        })
                        
                        app.logger.info(f"✓ Extracted attachment: {filename} ({mime_type})")
                        
                    except Exception as e:
                        app.logger.error(f"Failed to download attachment {filename}: {e}")
    
    # Process message parts
    if 'parts' in payload:
        process_parts(payload['parts'])
    
    return attachments


def upload_attachment_to_servicenow(sys_id: str, filename: str, file_data: str, content_type: str) -> bool:
    """
    Upload an attachment to a ServiceNow incident.
    
    Args:
        sys_id: ServiceNow incident sys_id
        filename: Name of the file
        file_data: Base64 encoded file data
        content_type: MIME type of the file
    
    Returns:
        True if successful, False otherwise
    """
    if not SN_INSTANCE or not SN_USER or not SN_PASSWORD:
        app.logger.error("ServiceNow credentials missing")
        return False
    
    base_url = f"https://{SN_INSTANCE}.service-now.com"
    url = f"{base_url}/api/now/attachment/file"
    
    # Decode base64 to binary
    try:
        # Add padding if needed
        file_data_padded = file_data + '=' * (-len(file_data) % 4)
        file_bytes = base64.urlsafe_b64decode(file_data_padded)
    except Exception as e:
        app.logger.error(f"Failed to decode attachment data: {e}")
        return False
    
    # Set parameters for the attachment
    params = {
        'table_name': 'incident',
        'table_sys_id': sys_id,
        'file_name': filename
    }
    
    headers = {
        'Content-Type': content_type,
        'Accept': 'application/json'
    }
    
    try:
        app.logger.info(f"Uploading attachment: {filename} to incident {sys_id}")
        
        response = requests.post(
            url,
            params=params,
            headers=headers,
            data=file_bytes,
            auth=(SN_USER, SN_PASSWORD),
            timeout=60  # Longer timeout for file uploads
        )
        
        response.raise_for_status()
        
        result = response.json().get('result', {})
        app.logger.info(f"✓ Successfully uploaded attachment: {filename} (sys_id: {result.get('sys_id')})")
        return True
        
    except requests.exceptions.HTTPError as e:
        app.logger.error(f"HTTP error uploading attachment {filename}: {e}")
        app.logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return False
    except Exception as e:
        app.logger.error(f"Error uploading attachment {filename}: {e}")
        return False


def process_and_upload_attachments(service, msg_id: str, payload: dict, incident_sys_id: str) -> dict:
    """
    Extract attachments from email and upload them to ServiceNow incident.
    Returns:
        Dict with 'count', 'successful', 'failed' keys """
    attachments = extract_attachments(service, msg_id, payload)
    
    if not attachments:
        return {'count': 0, 'successful': 0, 'failed': 0}
    
    app.logger.info(f"Found {len(attachments)} attachment(s) in email")
    
    successful = 0
    failed = 0
    
    for attachment in attachments:
        filename = attachment['filename']
        mime_type = attachment['mimeType']
        file_data = attachment['data']
        
        # Skip very large files (>10MB)
        if attachment.get('size', 0) > 10 * 1024 * 1024:
            app.logger.warning(f"Skipping large attachment: {filename} ({attachment['size']} bytes)")
            failed += 1
            continue
        
        success = upload_attachment_to_servicenow(
            incident_sys_id,
            filename,
            file_data,
            mime_type
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    return {
        'count': len(attachments),
        'successful': successful,
        'failed': failed
    }

# ---------------- BE SAFE PASSWORD RESET DETECTION ---------------- #
def is_besafe_password_reset(subject: str, body: str) -> bool:
    """
    Check if the email is about BE SAFE password reset.
    Checks both subject and body for BE SAFE related keywords.
    """
    if not subject and not body:
        return False
    
    # Combine subject and body for checking
    combined_text = f"{subject or ''} {body or ''}".lower()
    
    # Check if any BE SAFE keyword is present
    for keyword in BESAFE_KEYWORDS:
        if keyword.lower() in combined_text:
            app.logger.info(f"BE SAFE keyword detected: '{keyword}' in email")
            return True
    
    return False

def get_besafe_password_reset_template() -> str:
    """Return the template response for BE SAFE password reset requests."""
    return """Thank you for contacting us regarding BE SAFE password reset.
For Bsafe reset, Application password reset, MFA reset, or PingID reset, please contact ITSC at +442085624000.

Please note that resets will be performed only after successful validation through security questions and video verification via Microsoft Teams, using a valid Company ID card.

If the user experiences any issues with Microsoft Teams or the camera, a colleague or line manager may assist in completing the validation process.
Users attempting to reset their password must be connected to the BA network.
If you are working from home, please connect to the VPN, or visit the office or terminal to access the BA network.
 
Regards
ITSC """


    
def get_cc_addresses(headers: list) -> list:
    """Extract all CC email addresses from headers + extreme debug."""
    cc_header = get_header(headers, "Cc", default="")
    if not cc_header:
        cc_header = get_header(headers, "CC", default="")
    if not cc_header:
        cc_header = get_header(headers, "cc", default="")

    print("DEBUG_CC_RAW_HEADER     :", repr(cc_header))           # ← shows exact string with quotes
    print("DEBUG_CC_HEADER_LENGTH  :", len(cc_header))
    print("DEBUG_CC_HEADER_TYPE    :", type(cc_header))

    # Try multiple extraction patterns
    patterns = [
        r'[\w\.-]+@[\w\.-]+',                           # your current
        r'<([\w\.-]+@[\w\.-]+)>',                       # <email@domain.com>
        r'([\w\.-]+@[\w\.-]+)',                         # without <>
    ]

    all_emails = set()
    for pat in patterns:
        found = re.findall(pat, cc_header)
        all_emails.update(found)

    result = [email.lower().strip() for email in all_emails if email]

    print("DEBUG_CC_EXTRACTED_EMAILS :", result)
    print("DEBUG_CC_SKIP_LIST        :", CC_SKIP_ADDRESSES)
    print("DEBUG_CC_MATCH_CHECK      :",
          any(skip in result for skip in CC_SKIP_ADDRESSES))

    return result

def _extract_domain(email: str) -> str:
    """Return the domain part of an email address, normalized to lowercase."""
    if not email:
        return ""
    m = re.search(r'[\w\.-]+@([\w\.-]+)', email)
    domain = m.group(1) if m else ""
    return domain.lower()

def is_nonba_domain(email: str) -> bool:
    """Check if the email is from a NONBA domain (configured in .env)."""
    if not email or not NONBA_DOMAINS:
        return False
    domain = _extract_domain(email).lower()
    return domain in NONBA_DOMAINS

def is_blocked_sender(sender_header: str) -> bool:
    """Check if sender's domain matches the BLOCKED_DOMAINS list."""
    if not BLOCKED_DOMAINS:
        return False
    dom = _extract_domain(sender_header)
    if not dom:
        return False

    for pattern in BLOCKED_DOMAINS:
        if pattern.startswith("*."):
            base = pattern[2:]
            if dom.endswith("." + base):
                return True
        else:
            if dom == pattern:
                return True
    return False

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def decode_base64url(data: Optional[str]) -> str:
    if not data:
        return ""
    data += "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    except Exception:
        return ""

def strip_html(html: str) -> str:
    """Remove HTML tags but PRESERVE line breaks and formatting."""
    if not html:
        return ""
    
    # Convert HTML line breaks to actual newlines BEFORE stripping tags
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)
    html = re.sub(r'<br/>', '\n', html, flags=re.I)
    html = re.sub(r'</p>', '\n\n', html, flags=re.I)
    html = re.sub(r'</div>', '\n', html, flags=re.I)
    html = re.sub(r'</tr>', '\n', html, flags=re.I)
    html = re.sub(r'</li>', '\n', html, flags=re.I)
    
    # Remove all other HTML tags
    html = re.sub(r"<.*?>", "", html, flags=re.S)
    html = re.sub(r"&lt;.*?&gt;", "", html, flags=re.S)
    html = re.sub(r"&amp;lt;.*?&amp;gt;", "", html, flags=re.S)
    html = re.sub(r"&amp;amp;lt;.*?&amp;amp;gt;", "", html, flags=re.S)
    
    # Clean HTML entities
    html = re.sub(r'&nbsp;', ' ', html)
    html = re.sub(r'&quot;', '"', html)
    html = re.sub(r'&#39;', "'", html)
    html = re.sub(r'&amp;', '&', html)
    html = re.sub(r'&lt;', '<', html)
    html = re.sub(r'&gt;', '>', html)
    
    # DON'T collapse whitespace - preserve line breaks
    # Only collapse multiple spaces on the SAME line
    lines = html.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove excessive spaces within a line, but keep the line structure
        cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
        cleaned_lines.append(cleaned_line)
    
    # Join with newlines preserved
    result = '\n'.join(cleaned_lines)
    
    # Only remove excessive blank lines (4+), keep paragraph structure
    result = re.sub(r'\n\n\n\n+', '\n\n\n', result)
    
    return result.strip()

def extract_template_fields(email_body: str) -> dict:
    """
    Extract key fields from template response for short description.
    Returns dict with 'location' and 'application_name'.
    """
    if not email_body:
        return {"location": None, "application_name": None}
    
    body_lower = email_body.lower()
    lines = email_body.split('\n')
    
    location = None
    application_name = None
    
    # Patterns to match location
    location_patterns = [
        r'location\s*[:\-]\s*(.+?)(?:\n|$)',
        r'location\s*[:\-]\s*(.+?)(?:\||$)',
        r'location\s*(.+?)(?:\n|$)',
    ]
    
    # Patterns to match application name
    app_patterns = [
        r'impacted\s+application\s+name\s*[:\-]\s*(.+?)(?:\n|$)',
        r'application\s+name\s*[:\-]\s*(.+?)(?:\n|$)',
        r'application\s*[:\-]\s*(.+?)(?:\n|$)',
    ]
    
    # Extract location
    for pattern in location_patterns:
        match = re.search(pattern, email_body, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip()
            # Clean up common placeholders
            if extracted and not re.match(r'^\(.*\)$', extracted) and len(extracted) > 2:
                # Remove common placeholder text
                if 'terminal' not in extracted.lower() or len(extracted.split()) > 1:
                    location = extracted
                    break
    
    # Extract application name
    for pattern in app_patterns:
        match = re.search(pattern, email_body, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip()
            # Clean up and validate
            if extracted and len(extracted) > 2 and extracted.lower() not in ['n/a', 'na', 'none', '-']:
                application_name = extracted
                break
    
    # Clean extracted values
    if location:
        # Remove extra whitespace and limit length
        location = re.sub(r'\s+', ' ', location).strip()[:50]
    
    if application_name:
        # Remove extra whitespace and limit length
        application_name = re.sub(r'\s+', ' ', application_name).strip()[:50]
    
    return {
        "location": location,
        "application_name": application_name
    }

def extract_new_reply_content(email_body: str) -> str:
    """Extract ONLY the most recent reply from an email thread while preserving EXACT line breaks."""
    if not email_body:
        return ""
    
    # Patterns to identify where quoted/previous content starts
    quote_patterns = [
        r'\n\s*On\s+.+?wrote:',
        r'\n\s*From:\s*.+\n\s*Sent:',
        r'\n\s*-{3,}\s*Original Message\s*-{3,}',
        r'\n\s*-{3,}\s*Forwarded message\s*-{3,}',
        r'\n\s*Begin forwarded message:',
        r'\n\s*━{3,}',
        r'\n\s*═{3,}',
        r'\n\s*_{3,}',
        r'\n>{1,}\s*.+',
        r'\n\s+wrote:',
        r'\n\s*<.+@.+>\s+wrote:',
    ]
    
    # Find the earliest position where quoted content starts
    earliest_pos = len(email_body)
    
    for pattern in quote_patterns:
        match = re.search(pattern, email_body, re.IGNORECASE | re.MULTILINE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
    
    # Extract only the new content before any quoted material
    new_content = email_body[:earliest_pos].strip()
    
    # Remove lines that start with quote markers
    lines = new_content.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that are clearly quotes
        if re.match(r'^\s*[>|]+', line):
            continue
        if re.match(r'^\s*&gt;', line):
            continue
        cleaned_lines.append(line)
    
    new_content = '\n'.join(cleaned_lines)
    
    # Clean HTML entities
    new_content = re.sub(r'&quot;', '"', new_content)
    new_content = re.sub(r'&#39;', "'", new_content)
    new_content = re.sub(r'&amp;', '&', new_content)
    new_content = re.sub(r'&lt;', '<', new_content)
    new_content = re.sub(r'&gt;', '>', new_content)
    
    # CRITICAL: DO NOT remove any line breaks
    # Only remove excessive blank lines (4+ consecutive), keeping paragraph breaks
    new_content = re.sub(r'\n\n\n\n+', '\n\n\n', new_content)
    
    return new_content.strip()

def build_enhanced_short_description(original_subject: str, template_fields: dict) -> str:
    """
    Build short description: original subject + location + application name.
    Format: "Original Subject | Location: X | App: Y"
    """
    # Clean the original subject
    cleaned_subject = clean_subject_for_short_description(original_subject)
    
    if not cleaned_subject:
        cleaned_subject = "Support Request"
    
    parts = [cleaned_subject]
    
    # Add location if available
    location = template_fields.get("location")
    if location:
        parts.append(f"Location: {location}")
    
    # Add application name if available
    app_name = template_fields.get("application_name")
    if app_name:
        parts.append(f"App: {app_name}")
    
    # Join with pipe separator
    enhanced_description = " | ".join(parts)
    
    # Ensure it fits within ServiceNow's short_description limit (160 chars)
    if len(enhanced_description) > 160:
        # Try to trim gracefully
        if len(parts) > 1:
            # Recalculate with shorter versions
            short_subject = cleaned_subject[:80]
            short_location = location[:30] if location else None
            short_app = app_name[:30] if app_name else None
            
            new_parts = [short_subject]
            if short_location:
                new_parts.append(f"Loc: {short_location}")
            if short_app:
                new_parts.append(f"App: {short_app}")
            
            enhanced_description = " | ".join(new_parts)[:160]
        else:
            enhanced_description = enhanced_description[:160]
    
    return enhanced_description


def clean_user_reply(email_body: str, incident_number: Optional[str] = None) -> str:
    """Clean user reply by extracting only the newest content, preserving line breaks."""
    new_content = extract_new_reply_content(email_body)
    
    if not new_content:
        return ""
    
    # Remove our automated response content if present
    if incident_number:
        # Remove incident creation confirmation
        pattern = rf'Thank you!.*?logged as incident\s+{re.escape(incident_number)}.*?Support Team'
        new_content = re.sub(pattern, '', new_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove template sections that we sent
        template_removal_patterns = [
            r'═{3,}.*?═{3,}',
            r'─{3,}.*?',
            r'Simply reply to this email.*?filled template',
            r'Our support team will review.*?Support Team',
            r'Thank you for contacting us!.*?Support Team',
        ]
        
        for pattern in template_removal_patterns:
            new_content = re.sub(pattern, '', new_content, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Remove email signature footers
    footer_patterns = [
        r'\n\s*Sent from my iPhone.*',
        r'\n\s*Sent from my Android.*',
        r'\n\s*Get Outlook for.*',
        r'\n\s*Sent from Mail for Windows.*',
        r'\n\s*Sent from Yahoo Mail.*',
        r'\n\s*Sent from Samsung.*',
    ]
    
    for pattern in footer_patterns:
        new_content = re.sub(pattern, '', new_content, flags=re.IGNORECASE)
    
    # CRITICAL: Only collapse 3+ consecutive blank lines
    new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
    new_content = new_content.strip()
    
    if len(new_content) < 10:
        return "(User reply - content extracted)"
    
    return new_content

def extract_body(payload: dict) -> str:
    """
    Extract text body from Gmail message payload with multiple fallback methods.
    CRITICAL: This function MUST return content or attachments won't help.
    """
    app.logger.info("=" * 70)
    app.logger.info("EXTRACT_BODY - DEBUG START")
    
    body = ""
    
    # METHOD 1: Try to extract from parts (multipart emails)
    if "parts" in payload:
        app.logger.info("✓ Email has 'parts' - multipart email")
        parts = payload.get("parts", [])
        app.logger.info(f"Number of parts: {len(parts)}")
        
        for idx, part in enumerate(parts):
            mime_type = part.get("mimeType", "")
            app.logger.info(f"Part {idx}: mimeType = {mime_type}")
            
            # Try direct body data
            if part.get("mimeType") in ["text/plain", "text/html"]:
                data = part.get("body", {}).get("data")
                if data:
                    decoded = decode_base64url(data)
                    app.logger.info(f"✓ Part {idx} has body data: {len(decoded)} chars")
                    if decoded and len(decoded) > len(body):
                        body = decoded
            
            # Check nested parts (for complex multipart)
            if "parts" in part:
                app.logger.info(f"Part {idx} has nested parts")
                for nested_idx, nested_part in enumerate(part.get("parts", [])):
                    nested_mime = nested_part.get("mimeType", "")
                    app.logger.info(f"  Nested part {nested_idx}: mimeType = {nested_mime}")
                    
                    if nested_part.get("mimeType") in ["text/plain", "text/html"]:
                        data = nested_part.get("body", {}).get("data")
                        if data:
                            decoded = decode_base64url(data)
                            app.logger.info(f"✓ Nested part {nested_idx} has body data: {len(decoded)} chars")
                            if decoded and len(decoded) > len(body):
                                body = decoded
    
    # METHOD 2: Try direct body (simple emails)
    if not body:
        app.logger.info("Trying direct body extraction")
        direct_data = payload.get("body", {}).get("data")
        if direct_data:
            body = decode_base64url(direct_data)
            app.logger.info(f"✓ Direct body data: {len(body)} chars")
    
    # METHOD 3: Last resort - check payload.body.data
    if not body:
        app.logger.info("Trying payload.body as last resort")
        if "body" in payload and "data" in payload["body"]:
            body = decode_base64url(payload["body"]["data"])
            app.logger.info(f"✓ Payload.body.data: {len(body)} chars")
    
    # Strip HTML if we got content
    if body:
        app.logger.info(f"Before strip_html: {len(body)} chars")
        body = strip_html(body)
        app.logger.info(f"After strip_html: {len(body)} chars")
    
    app.logger.info("=" * 70)
    app.logger.info("EXTRACT_BODY - RESULT")
    app.logger.info(f"Final body length: {len(body)} chars")
    if body:
        app.logger.info(f"First 200 chars: {body[:200]}")
    else:
        app.logger.error("❌ BODY IS EMPTY - THIS WILL CAUSE DESCRIPTION TO BE EMPTY!")
    app.logger.info("=" * 70)
    
    return body

def get_header(headers: list, name: str, default: str = "") -> str:
    """Safely fetch a header value by name (case-insensitive)."""
    target = name.lower()
    for h in headers or []:
        if h.get("name", "").lower() == target:
            return h.get("value", default)
    return default

def _normalize_for_embedding(s: str) -> str:
    s = s or ""
    return unicodedata.normalize("NFKC", s).strip()

def normalize_subject(subj: str) -> str:
    """Normalize subject by stripping Re:/Fwd:/FW: prefixes and collapsing whitespace."""
    s = (subj or "").strip()
    s = re.sub(r'^(?:\s*(re|fwd|fw)\s*[:\-]\s*)+', '', s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def normalize_sender_addr(sender_header: str) -> str:
    """Extract and normalize sender email address."""
    m = re.search(r'[\w\.-]+@[\w\.-]+', sender_header or "")
    return (m.group(0).lower() if m else (sender_header or "").lower())

def extract_email_parts(email_address: str) -> dict:
    """
    Extract email address and generate firstname from email.
    For NONBA domains (configured in .env), set firstname as "NONBA".
    Example: chat_bot@gmail.com -> {email: 'chat_bot@gmail.com', firstname: 'chatbot'}
    Example: chatbot2781@outlook.com -> {email: 'chatbot2781@outlook.com', firstname: 'NONBA'}
    """
    if not email_address:
        return {"email": "", "firstname": ""}
    
    # Extract email from "Name <email@domain.com>" format
    match = re.search(r'[\w\.-]+@[\w\.-]+', email_address)
    if not match:
        return {"email": "", "firstname": ""}
    
    email = match.group(0).lower()
    
    # Check if it's a NONBA domain
    if is_nonba_domain(email):
        return {
            "email": email,
            "firstname": "NONBA"
        }
    
    # Extract username part before @ for non-NONBA emails
    username = email.split('@')[0]
    
    # Convert username to firstname: remove special chars, underscores, dots
    firstname = re.sub(r'[._-]', '', username).lower()
    
    return {
        "email": email,
        "firstname": firstname
    }
def get_or_create_servicenow_caller(sender_email: str) -> Optional[str]:
    """
    Get existing caller sys_id by email, or create new caller if not found.
    For NONBA domains (configured in .env), sets firstname as "NONBA".
    Returns the caller sys_id.
    """
    if not SN_INSTANCE or not SN_USER or not SN_PASSWORD:
        app.logger.error("ServiceNow credentials missing")
        return None
    
    email_parts = extract_email_parts(sender_email)
    email = email_parts["email"]
    firstname = email_parts["firstname"]
    
    if not email or not firstname:
        app.logger.warning(f"Could not extract valid email/firstname from: {sender_email}")
        return None
    
    base_url = f"https://{SN_INSTANCE}.service-now.com"
    
    # First, try to find existing user by email
    search_url = f"{base_url}/api/now/table/sys_user"
    search_params = {
        "sysparm_query": f"email={email}",
        "sysparm_fields": "sys_id,email,first_name,user_name",
        "sysparm_limit": "1"
    }
    
    try:
        app.logger.info(f"Searching for user with email: {email}")
        resp = requests.get(search_url, params=search_params, auth=(SN_USER, SN_PASSWORD), timeout=30)
        resp.raise_for_status()
        results = resp.json().get("result", [])
        
        if results:
            # User exists, return sys_id
            sys_id = results[0].get("sys_id")
            app.logger.info(f"✓ Found existing caller: {email} with sys_id: {sys_id}")
            return sys_id
        
        # User doesn't exist, create new one
        is_nonba = is_nonba_domain(email)
        app.logger.info(f"No existing user found. Creating new caller for email: {email} (NONBA: {is_nonba}, firstname: {firstname})")
        
        create_url = f"{base_url}/api/now/table/sys_user"
        create_payload = {
            "email": email,
            "first_name": firstname,
            "last_name": "EmailBot",  # Required field
            "user_name": email,
            "active": "true",  # String instead of boolean
            "source": "email_bot_nonba" if is_nonba else "email_bot"
        }
        
        app.logger.info(f"Creating user with payload: {create_payload}")
        
        create_resp = requests.post(
            create_url, 
            json=create_payload, 
            auth=(SN_USER, SN_PASSWORD), 
            timeout=30,
            headers={"Content-Type": "application/json", "Accept": "application/json"}
        )
        
        if not create_resp.ok:
            app.logger.error(f"Failed to create user. Status: {create_resp.status_code}, Response: {create_resp.text}")
            return None
            
        create_resp.raise_for_status()
        
        new_user = create_resp.json().get("result", {})
        sys_id = new_user.get("sys_id")
        
        if sys_id:
            app.logger.info(f"✓ Created new caller: {email} with sys_id: {sys_id}, firstname: {firstname}")
        else:
            app.logger.error(f"User creation response missing sys_id: {create_resp.text}")
            
        return sys_id
        
    except requests.exceptions.HTTPError as e:
        app.logger.error(f"HTTP Error getting/creating caller for {email}: {e}")
        app.logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return None
    except Exception as e:
        app.logger.error(f"Error getting/creating caller for {email}: {e}")
        return None
'''    
def get_or_create_servicenow_caller(sender_email: str) -> Optional[str]:
    """
    Get existing caller sys_id by email, or create new caller if not found.
    For NONBA domains (configured in .env), sets firstname as "NONBA".
    Returns the caller sys_id.
    """
    if not SN_INSTANCE or not SN_USER or not SN_PASSWORD:
        app.logger.error("ServiceNow credentials missing")
        return None
    
    email_parts = extract_email_parts(sender_email)
    email = email_parts["email"]
    firstname = email_parts["firstname"]
    
    if not email or not firstname:
        app.logger.warning(f"Could not extract valid email/firstname from: {sender_email}")
        return None
    
    base_url = f"https://{SN_INSTANCE}.service-now.com"
    
    # First, try to find existing user by email
    search_url = f"{base_url}/api/now/table/sys_user"
    search_params = {
        "sysparm_query": f"email={email}",
        "sysparm_fields": "sys_id,email,first_name,user_name",
        "sysparm_limit": "1"
    }
    
    try:
        app.logger.info(f"Searching for user with email: {email}")
        resp = requests.get(search_url, params=search_params, auth=(SN_USER, SN_PASSWORD), timeout=30)
        resp.raise_for_status()
        results = resp.json().get("result", [])
        
        if results:
            # User exists, return sys_id
            sys_id = results[0].get("sys_id")
            app.logger.info(f"✓ Found existing caller: {email} with sys_id: {sys_id}")
            return sys_id
        
        # User doesn't exist, create new one
        is_nonba = is_nonba_domain(email)
        app.logger.info(f"No existing user found. Creating new caller for email: {email} (NONBA: {is_nonba}, firstname: {firstname})")
        
        create_url = f"{base_url}/api/now/table/sys_user"
        create_payload = {
        "email": email,
        "first_name": firstname,
        "last_name": "EmailBot",  # ← ADD THIS
        "user_name": email,
        "active": "true",  # ← String not boolean
        "source": "email_bot_nonba" if is_nonba else "email_bot"
        }
        
        app.logger.info(f"Creating user with payload: {create_payload}")
        
           # Also add headers to the request:
        create_resp = requests.post(
            create_url, 
            json=create_payload, 
            auth=(SN_USER, SN_PASSWORD), 
            timeout=30,
            headers={"Content-Type": "application/json", "Accept": "application/json"}  # ← ADD THIS
        )
        
        if not create_resp.ok:
            app.logger.error(f"Failed to create user. Status: {create_resp.status_code}, Response: {create_resp.text}")
            return None
            
        create_resp.raise_for_status()
        
        new_user = create_resp.json().get("result", {})
        sys_id = new_user.get("sys_id")
        
        if sys_id:
            app.logger.info(f"✓ Created new caller: {email} with sys_id: {sys_id}, firstname: {firstname}")
        else:
            app.logger.error(f"User creation response missing sys_id: {create_resp.text}")
            
        return sys_id
        
    except requests.exceptions.HTTPError as e:
        app.logger.error(f"HTTP Error getting/creating caller for {email}: {e}")
        app.logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return None
    except Exception as e:
        app.logger.error(f"Error getting/creating caller for {email}: {e}")
        return None
'''

# ---------------- SUBJECT HELPERS ---------------- #
INC_SUBJECT_FORMAT_NEW = os.getenv("INC_SUBJECT_FORMAT_NEW", "Re:{subject} -[{number}]")
INC_SUBJECT_FORMAT_STATUS = os.getenv("INC_SUBJECT_FORMAT_STATUS", "Updates on incident-({number})")
TEMPLATE_REQUEST_SUBJECT = os.getenv("TEMPLATE_REQUEST_SUBJECT", "Re: {subject} - Additional Information Required")

def format_new_incident_subject(number: str, subject: str) -> str:
    """Subject for newly created incident reply."""
    number = number or "INC-UNKNOWN"
    subject = subject or ""
    try:
        return INC_SUBJECT_FORMAT_NEW.format(number=number, subject=subject)
    except Exception:
        return f"Re:[{number}] {subject}"

def format_status_subject(number: str) -> str:
    """Subject for status replies."""
    number = number or "INC-UNKNOWN"
    try:
        return INC_SUBJECT_FORMAT_STATUS.format(number=number)
    except Exception:
        return f"Updates on incident-({number})"

def format_template_request_subject(subject: str) -> str:
    """Subject for template request emails."""
    subject = subject or "(No Subject)"
    try:
        return TEMPLATE_REQUEST_SUBJECT.format(subject=subject)
    except Exception:
        return f"Re: {subject} - Additional Information Required"

INC_NUMBER_REGEX = re.compile(r"(INC\d{4,})", re.I)

def extract_inc_number_from_subject(subject: str) -> Optional[str]:
    """Extract INC number (e.g., INC0012345) from subject if present."""
    if not subject:
        return None
    m = INC_NUMBER_REGEX.search(subject)
    return m.group(1).upper() if m else None

# ---------------- TEMPLATE DETECTION ---------------- #
TEMPLATE_MARKER = "[TEMPLATE_REQUEST]"

def is_template_response(email_body: str, subject: str) -> bool:
    """
    Check if this email is a response to a template request.
    IMPROVED: Detects actual filled content, not just field labels.
    """
    if not email_body:
        app.logger.info("No email body - not a template response")
        return False
    
    # PRIORITY 1: Check if subject contains template request marker
    if "additional information required" in (subject or "").lower():
        app.logger.info("✓ Subject contains 'additional information required' - IS template response")
        return True
    
    # PRIORITY 2: Check for actual FILLED content (not just field labels)
    # These patterns look for field:VALUE pairs with actual content
    filled_field_patterns = [
        r'user\s*name[:/\-\s]*[a-zA-Z0-9@.]{2,}',  # At least 2 chars of content
        r'contact\s*number[:/\-\s]*[\d\s+()-]{7,}',  # At least 7 digits
        r'location[:/\-\s]*[a-zA-Z]{3,}',  # At least 3 letters
        r'application\s*name[:/\-\s]*[a-zA-Z]{3,}',  # At least 3 letters
    ]
    
    body_lower = email_body.lower()
    filled_matches = 0
    
    for pattern in filled_field_patterns:
        if re.search(pattern, body_lower, re.IGNORECASE):
            filled_matches += 1
            app.logger.info(f"✓ Found filled field matching: {pattern}")
    
    # If user has filled at least 3 fields with actual content, it's a template response
    if filled_matches >= 3:
        app.logger.info(f"✓ TEMPLATE RESPONSE DETECTED: {filled_matches} filled fields found")
        return True
    
    # Fallback check: Look for EMPTY field labels (user hasn't filled template yet)
    empty_field_patterns = [
        r'user\s*name\s*[:/\-]\s*$',  # Field label with nothing after
        r'contact\s*number\s*[:/\-]\s*$',
        r'location\s*[:/\-]\s*$',
    ]
    
    empty_matches = 0
    for pattern in empty_field_patterns:
        if re.search(pattern, email_body, re.IGNORECASE | re.MULTILINE):
            empty_matches += 1
    
    # If we see 2+ empty field labels, this is NOT a filled response
    if empty_matches >= 2:
        app.logger.info(f"✗ NOT a template response: {empty_matches} EMPTY field labels detected")
        return False
    
    app.logger.info(f"✗ NOT a template response: only {filled_matches} filled fields (need 3+)")
    return False

# ---------------- PURE SIMILARITY DECISION ---------------- #
def generate_response(email_body: str, threshold: float = 0.60):
    """Similarity-only decision with template request fallback."""
    text = (email_body or "").strip()
    if not text:
        return {"action": "REQUEST_TEMPLATE", "similarity": 0.0}

    if not (model and faiss_index is not None and df is not None):
        return {"action": "REQUEST_TEMPLATE", "similarity": 0.0}

    q = _normalize_for_embedding(text)
    q_emb = model.encode([q])
    q_emb = np.asarray(q_emb, dtype=np.float32)
    q_emb_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    TOP_K = 5
    D, I = faiss_index.search(q_emb_norm, TOP_K)

    best_idx = None
    best_sim = -1.0

    for rank in range(min(TOP_K, len(I[0]))):
        try:
            neighbor_vec = faiss_index.reconstruct(int(I[0][rank]))
            neighbor_vec = np.asarray(neighbor_vec, dtype=np.float32)
            neighbor_norm = neighbor_vec / (np.linalg.norm(neighbor_vec) + 1e-12)
            sim = float(np.dot(q_emb_norm[0], neighbor_norm))
        except Exception:
            dist = float(D[0][rank])
            sim = float(1.0 / (1.0 + max(0.0, dist)))

        if sim > best_sim:
            best_sim = sim
            best_idx = int(I[0][rank])

    if best_sim < threshold or best_idx is None:
        return {"action": "REQUEST_TEMPLATE", "similarity": best_sim}

    response_text = str(df.iloc[best_idx]["email_response"])
    return {
        "action": "AUTO_RESPONSE",
        "similarity": best_sim,
        "response": response_text,
    }

# ---------------- SEND EMAIL (UPDATED) ---------------- #
def send_email(service, to_email: str, subject: str, body: str, is_html: bool = False):
    """Send email with optional HTML support."""
    if is_html:
        message = MIMEText(body, 'html')
    else:
        message = MIMEText(body)
    
    message["to"] = to_email
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()

# ---------------- SERVICENOW INTEGRATION ---------------- #
SN_INSTANCE = os.getenv("SN_INSTANCE")
SN_USER = os.getenv("SN_USER")
SN_PASSWORD = os.getenv("SN_PASSWORD")

SN_FIELDS = "number,short_description,sys_id,priority,opened_at"
SN_STATUS_FIELDS = "number,short_description,sys_id,priority,state,opened_at,sys_updated_on,assignment_group"

def _safe_trim(text: str, max_len: int = 160) -> str:
    """Trim to ServiceNow short_description-friendly length."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

def create_servicenow_incident(
    subject: str,
    body: str,
    priority: str = "5",
    correlation_id: Optional[str] = None,
    channel: str = "email",
    assignment_group: str = "SVC DESC",
    caller_email: Optional[str] = None,
    template_fields: Optional[dict] = None  # NEW PARAMETER
):
    """Create incident with caller information and enhanced short description."""
    if not SN_INSTANCE or not SN_USER or not SN_PASSWORD:
        raise RuntimeError("ServiceNow credentials missing: set SN_INSTANCE, SN_USER, SN_PASSWORD env vars.")

    base_url = f"https://{SN_INSTANCE}.service-now.com"
    url = f"{base_url}/api/now/table/incident"

    # Build enhanced short description if template fields provided
    if template_fields:
        short_desc = build_enhanced_short_description(subject, template_fields)
    else:
        short_desc = _safe_trim(clean_subject_for_short_description(subject) or "No subject")

    # Build description with debug logging
    description_value = body or "No description"

    app.logger.info("=" * 70)
    app.logger.info("CREATE_SERVICENOW_INCIDENT - PAYLOAD DEBUG")
    app.logger.info(f"short_description: {short_desc}")
    app.logger.info(f"description length: {len(description_value)} chars")
    app.logger.info(f"description content (first 300 chars):")
    app.logger.info(description_value[:300])
    app.logger.info("=" * 70)

    payload = {
        "short_description": short_desc,
        "description": description_value,
        "priority": str(priority or "5"),
        "contact_type": channel,
    }
    
    # Add caller_id
    if caller_email:
        app.logger.info(f"Attempting to set caller for email: {caller_email}")
        try:
            caller_sys_id = get_or_create_servicenow_caller(caller_email)
            if caller_sys_id:
                payload["caller_id"] = caller_sys_id
                app.logger.info(f"✓ Successfully added caller_id to payload: {caller_sys_id}")
            else:
                app.logger.warning(f"⚠ Failed to get caller_sys_id for: {caller_email}")
        except Exception as e:
            app.logger.error(f"✗ Exception while setting caller_id: {e}")
    else:
        app.logger.warning("⚠ No caller_email provided to create_servicenow_incident")
    
    if correlation_id:
        payload["correlation_id"] = correlation_id
    
    if assignment_group:
        payload["assignment_group"] = assignment_group

    app.logger.info(f"Creating incident with enhanced short_description: {short_desc}")
    app.logger.info(f"Template fields: {template_fields}")
    
    try:
        resp = requests.post(url, json=payload, auth=(SN_USER, SN_PASSWORD), timeout=30)
        
        if not resp.ok:
            app.logger.error(f"ServiceNow incident creation failed. Status: {resp.status_code}")
            app.logger.error(f"Response: {resp.text}")
            resp.raise_for_status()
            
        result = resp.json().get("result", {}) or {}

        if "number" not in result or "sys_id" not in result:
            app.logger.error(f"Unexpected ServiceNow response: {resp.text}")
            raise ValueError(f"Unexpected ServiceNow response: {resp.text}")
        
        # Verify caller was set
        if "caller_id" in result:
            app.logger.info(f"✓ Incident {result.get('number')} created with caller: {result.get('caller_id')}")
        else:
            app.logger.warning(f"⚠ Incident {result.get('number')} created but caller_id not in response")

        return result
        
    except Exception as e:
        app.logger.error(f"Exception creating ServiceNow incident: {e}")
        raise

def get_servicenow_incident(
    number: Optional[str] = None,
    sys_id: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> Optional[dict]:
    """Fetch incident by number, sys_id or correlation_id; returns dict or None."""
    if not SN_INSTANCE or not SN_USER or not SN_PASSWORD:
        raise RuntimeError("ServiceNow credentials missing: set SN_INSTANCE, SN_USER, SN_PASSWORD env vars.")

    base_url = f"https://{SN_INSTANCE}.service-now.com"
    url = f"{base_url}/api/now/v1/table/incident"

    params = {
        "sysparm_fields": SN_STATUS_FIELDS,
        "sysparm_exclude_reference_link": "true",
        "sysparm_display_value": "true",
        "sysparm_limit": "1"
    }

    query = []
    if sys_id:
        query.append(f"sys_id={sys_id}")
    elif number:
        query.append(f"number={number}")
    elif correlation_id:
        query.append(f"correlation_id={correlation_id}")

    if not query:
        return None

    params["sysparm_query"] = "^".join(query)

    resp = requests.get(url, params=params, auth=(SN_USER, SN_PASSWORD), timeout=30)
    resp.raise_for_status()
    items = resp.json().get("result", []) or []
    return items[0] if items else None

def post_incident_comment(sys_id: str, comment: str) -> dict:
    """Append a new Additional comments entry to an incident."""
    base_url = f"https://{SN_INSTANCE}.service-now.com"
    url = f"{base_url}/api/now/v1/table/incident/{sys_id}"
    payload = {"comments": comment or ""}
    params = {
        "sysparm_display_value": "true",
        "sysparm_exclude_reference_link": "true",
        "sysparm_fields": SN_STATUS_FIELDS,
    }
    resp = requests.patch(url, params=params, json=payload, auth=(SN_USER, SN_PASSWORD), timeout=30)
    resp.raise_for_status()
    return resp.json().get("result", {}) or {}

# ---------------- LOCAL INCIDENT MAPPING ---------------- #
INCIDENT_MAP_FILE = os.getenv("INCIDENT_MAP_FILE", "incident_map.pkl")
TEMPLATE_PENDING_FILE = os.getenv("TEMPLATE_PENDING_FILE", "template_pending.pkl")

incident_map_by_thread = {}
incident_map_by_subject = {}
template_pending = {}  # thread_id -> {"sender": ..., "subject": ..., "timestamp": ...}

def _load_incident_map():
    global incident_map_by_thread, incident_map_by_subject, template_pending
    try:
        with open(INCIDENT_MAP_FILE, "rb") as f:
            data = pickle.load(f)
            incident_map_by_thread = data.get("by_thread", {})
            incident_map_by_subject = data.get("by_subject", {})
            app.logger.info(f"Loaded incident map: {len(incident_map_by_thread)} threads, {len(incident_map_by_subject)} subjects.")
    except Exception:
        incident_map_by_thread = {}
        incident_map_by_subject = {}
    
    try:
        with open(TEMPLATE_PENDING_FILE, "rb") as f:
            template_pending = pickle.load(f)
            app.logger.info(f"Loaded template pending: {len(template_pending)} entries.")
    except Exception:
        template_pending = {}

def _save_incident_map():
    try:
        with open(INCIDENT_MAP_FILE, "wb") as f:
            pickle.dump({"by_thread": incident_map_by_thread, "by_subject": incident_map_by_subject}, f)
    except Exception as e:
        app.logger.warning(f"Failed to persist incident map: {e}")

def _save_template_pending():
    try:
        with open(TEMPLATE_PENDING_FILE, "wb") as f:
            pickle.dump(template_pending, f)
    except Exception as e:
        app.logger.warning(f"Failed to persist template pending: {e}")

_load_incident_map()

def _cache_incident(sender: str, subject: str, thread_id: Optional[str], inc: dict):
    """Persist incident mapping for later lookups."""
    sender_email = normalize_sender_addr(sender)
    subj_key = normalize_subject(subject)
    payload = {
        "number": inc.get("number"),
        "sys_id": inc.get("sys_id"),
        "short_description": inc.get("short_description"),
        "priority": inc.get("priority"),
        "thread_id": thread_id,
    }
    if thread_id:
        incident_map_by_thread[thread_id] = payload
    incident_map_by_subject[(sender_email, subj_key)] = payload
    _save_incident_map()

def _lookup_cached_incident(sender: str, subject: str, thread_id: Optional[str]) -> Optional[dict]:
    """Return cached mapping if present (without hitting ServiceNow)."""
    if thread_id and thread_id in incident_map_by_thread:
        return incident_map_by_thread.get(thread_id)
    sender_email = normalize_sender_addr(sender)
    subj_key = normalize_subject(subject)
    return incident_map_by_subject.get((sender_email, subj_key))

def _mark_template_pending(thread_id: str, sender: str, subject: str):
    """Mark a thread as waiting for template response."""
    template_pending[thread_id] = {
        "sender": sender,
        "subject": subject,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    _save_template_pending()

def _is_template_pending(thread_id: str) -> bool:
    """Check if a thread is waiting for template response."""
    return thread_id in template_pending

def _clear_template_pending(thread_id: str):
    """Clear template pending status after incident creation."""
    if thread_id in template_pending:
        del template_pending[thread_id]
        _save_template_pending()

# ---------------- STATUS EMAIL ---------------- #
def format_incident_status_reply(incident: dict, last_comment: Optional[dict] = None, include_form_link: Optional[str] = None) -> str:
    """Compose a user-friendly status reply."""
    number = incident.get("number", "UNKNOWN")
    short_desc = incident.get("short_description", "(No short description)")
    priority = incident.get("priority", "UNKNOWN")
    state = incident.get("state", "UNKNOWN")
    updated = incident.get("sys_updated_on", incident.get("opened_at", ""))
    assignment_group = incident.get("assignment_group", "")
    if isinstance(assignment_group, dict):
        assignment_group = assignment_group.get("display_value", "") or ""

    lines = [
        f"Status update for your existing incident {number}:",
        "",
        f"• Short description: {short_desc}",
        f"• State: {state}",
        f"• Priority: {priority}",
        f"• Last updated: {updated}",
    ]
    if assignment_group:
        lines.append(f"• Assignment group: {assignment_group}")

    if str(state).lower() == "on hold":
        lines += ["", "The incident is currently On Hold. The latest request from our team:"]
        if last_comment and last_comment.get("value"):
            lines.append(f"\"{last_comment['value']}\"")
            if last_comment.get("sys_created_on") or last_comment.get("sys_created_by"):
                meta = []
                if last_comment.get("sys_created_on"):
                    meta.append(f"on {last_comment['sys_created_on']}")
                if last_comment.get("sys_created_by"):
                    meta.append(f"by {last_comment['sys_created_by']}")
                if meta:
                    lines.append(f"({', '.join(meta)})")
        else:
            lines.append("(No latest comment found.)")
        lines += [
            "",
            "You can provide the requested information by using the form below:"
        ]
        if include_form_link:
            lines += [include_form_link]
        lines += [
            "",
            "Once you provide the details, we will add them to the incident and continue processing."
        ]
    else:
        lines += [
            "",
            "We are actively working on your request. If you have more details to add, just reply to this email."
        ]
    return "\n".join(lines)

def _get_gmail_service_from_session() -> Optional[any]:
    creds_json = session.get("credentials")
    if not creds_json:
        return None
    info = json.loads(creds_json)
    creds = Credentials.from_authorized_user_info(info, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        session["credentials"] = creds.to_json()
    return build("gmail", "v1", credentials=creds)

def set_incident_to_in_progress(sys_id: str) -> bool:
    """Update incident state to 'In Progress' (state = 2)."""
    base_url = f"https://{SN_INSTANCE}.service-now.com"
    url = f"{base_url}/api/now/v1/table/incident/{sys_id}"
    payload = {"state": "2"}
    params = {
        "sysparm_display_value": "true",
        "sysparm_exclude_reference_link": "true",
    }
    try:
        resp = requests.patch(url, params=params, json=payload, auth=(SN_USER, SN_PASSWORD), timeout=30)
        resp.raise_for_status()
        app.logger.info(f"Incident {sys_id} state updated to In Progress")
        return True
    except Exception as e:
        app.logger.warning(f"Failed to update incident {sys_id} to In Progress: {e}")
        return False

def get_recent_work_notes(sys_id: str, limit: int = 5) -> list:
    """Return a list of recent Work Notes (newest first) for an incident."""
    if limit < 1:
        limit = 1
    if limit > 10:
        limit = 10

    base_url = f"https://{SN_INSTANCE}.service-now.com"
    url = f"{base_url}/api/now/v1/table/sys_journal_field"
    params = {
        "sysparm_query": f"element_id={sys_id}^element=work_notes^ORDERBYDESCsys_created_on",
        "sysparm_limit": str(limit),
        "sysparm_fields": "value,sys_created_on,sys_created_by",
        "sysparm_display_value": "true",
        "sysparm_exclude_reference_link": "true",
    }
    try:
        resp = requests.get(url, params=params, auth=(SN_USER, SN_PASSWORD), timeout=30)
        resp.raise_for_status()
        items = resp.json().get("result", []) or []
        return items
    except Exception as e:
        app.logger.warning(f"Failed to fetch recent work notes for {sys_id}: {e}")
        return []

# ---------------- INFORMATION TEMPLATE (UPDATED WITH HTML TABLE) ---------------- #
def get_blocked_domain_template() -> str:
    """Return the template for blocked domain users with self-service instructions."""
    return """Hi,

Thank you for contacting us.

As we could see that you already have access for IT Service portal/ ServiceNow, Please log your queries directly through the IT Service Portal.

If you are facing any issues or problems, log it under the IT Service Portal -> FIX IT:

If your query is regarding a new requirement or services, log it under the  IT Service Portal -> REQUEST IT.

Thanks for your understanding and cooperation.

Regards,
ITSC """

def get_information_template() -> str:
    """Return the standard template for collecting user information in HTML table format."""
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
</head>
<body>
    <p>Thank you for contacting us! To help us resolve your issue efficiently, please provide the following information by replying to this email:</p>
    
    <h3 style="color: #2563eb; margin-top: 20px;">General Information (Applicable to All Issues)</h3>
    <br>
    
    <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; max-width: 800px; margin: 15px 0;">
        <tr>
            <td style="background-color: #f0f0f0; font-weight: bold; width: 45%;">User Name / ID / Email :*</td>
            <td style="width: 55%; background-color: #ffffff;"></td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0; font-weight: bold;">Contact Number:*</td>
            <td style="background-color: #ffffff;"></td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0; font-weight: bold;">Location:*</td>
            <td style="background-color: #ffffff; color: #666;">(Terminal / Building / Landside / Airside / Desk / Office etc.)</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0; font-weight: bold;">Issue Start Date:*</td>
            <td style="background-color: #ffffff;"></td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0; font-weight: bold;">Impacted Application Name:*</td>
            <td style="background-color: #ffffff;"></td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0; font-weight: bold;">Any Workaround Available?</td>
            <td style="background-color: #ffffff;"></td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0; font-weight: bold;">Previous Incident Reference (if any):</td>
            <td style="background-color: #ffffff;"></td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0; font-weight: bold;">Error Screenshot:</td>
            <td style="background-color: #ffffff; color: #666;">(Please attach)</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0; font-weight: bold;">Mirror ID (if access-related):</td>
            <td style="background-color: #ffffff;"></td>
        </tr>
    </table>
    
    <h3 style="color: #2563eb; margin-top: 25px;">Issue Description:*</h3>
    
    <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; max-width: 800px; margin: 15px 0;">
        <tr>
            <td style="min-height: 80px; background-color: #ffffff; color: #666;">[Please describe your issue in detail here]</td>
        </tr>
    </table>
    
    <hr style="border: none; border-top: 2px solid #e0e0e0; margin: 25px 0;">
    
    <p style="margin-top: 20px;">Simply reply to this email with the filled information above.
    * are mandatory Fields.
    Once we receive your information, we will create an incident and begin working on your request.</p>
    
    <p style="margin-top: 15px;">Thank you,<br/>
    <strong>Support Team</strong></p>
</body>
</html>"""

def get_cc_skip_template() -> str:
    """Return the BA IT Service Desk template for CC skip address emails."""
    return """Thank you for contacting the British Airways IT Service Desk. Please note that this mailbox is now managed via an automated bot.

We've identified that the IT Service Centre (ITSC) is not the primary recipient of this email, and therefore no action will be taken by the Service Desk.

If you require assistance, please use one of the alternative contact methods below:

═══════════════════════════════════════════════════════════

For Urgent Faults
(Incidents impacting or about to impact business‑critical services, affecting multiple services or a significant number of staff, and requiring immediate attention to prevent an operational or commercial disruption.)

Please call the Group IT Service Desk:
• UK: 020 856 24000
• Overseas: +44 208 562 4000

═══════════════════════════════════════════════════════════

For Non‑Urgent Faults or Requests for Service (RFS)
Please raise these via the Group IT Self‑Service Portal.

Using the portal ensures a faster, more efficient experience, allowing you to track progress and manage communication effectively.


You can also browse FAQs, user guides, and how‑to articles on the IT Service Portal for quick self‑help solutions to common issues.

Regards,
The IT Service Desk"""


def format_closed_incident_status_with_template(incident: dict) -> str:
    """
    Compose a status reply for closed incidents that includes the template request.
    This is sent when user replies to a closed incident.
    """
    number = incident.get("number", "UNKNOWN")
    short_desc = incident.get("short_description", "(No short description)")
    state = incident.get("state", "UNKNOWN")
    updated = incident.get("sys_updated_on", incident.get("opened_at", ""))
    
    # Get the information template HTML
    template_html = get_information_template()
    
    # Extract the body content from template
    body_match = re.search(r'<body[^>]*>(.*?)</body>', template_html, re.DOTALL | re.IGNORECASE)
    template_body_content = body_match.group(1) if body_match else template_html
    
    # Compose combined message
    combined_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
</head>
<body>
    <div style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
        <h3 style="color: #856404; margin-top: 0;">📋 Previous Incident Status</h3>
        <p>Your referenced incident <strong>{number}</strong> has been <strong>CLOSED</strong>.</p>
        <ul style="margin: 10px 0;">
            <li><strong>Short description:</strong> {short_desc}</li>
            <li><strong>State:</strong> {state}</li>
            <li><strong>Last updated:</strong> {updated}</li>
        </ul>
        <p style="margin-top: 15px;"><strong>Since this incident is closed, we'll create a NEW incident for your current request.</strong></p>
    </div>
    
    <hr style="border: none; border-top: 2px solid #e0e0e0; margin: 25px 0;">
    
    <div style="margin-top: 20px;">
        <h3 style="color: #2563eb;">🎫 Create New Incident</h3>
        <p><strong>Please provide the following information to create a new incident:</strong></p>
        {template_body_content}
    </div>
</body>
</html>"""
    
    return combined_html

def clean_template_response(email_body: str) -> str:
    """
    Clean template response while preserving ALL user content but removing our template.
    CRITICAL: This function MUST return only the user's filled content, not our template.
    """
    if not email_body:
        app.logger.error("❌ clean_template_response: email_body is EMPTY or None")
        return "(No content provided by user)"
    
    app.logger.info("=" * 70)
    app.logger.info("CLEAN_TEMPLATE_RESPONSE - START")
    app.logger.info(f"Input length: {len(email_body)} characters")
    app.logger.info(f"First 200 chars of input: {email_body[:200]}")
    app.logger.info("=" * 70)
    
    # Step 1: Convert literal \n to actual newlines
    original_body = email_body
    email_body = email_body.replace('\\n', '\n')
    if email_body != original_body:
        app.logger.info(f"✓ Converted literal \\n to newlines")
    
    # Step 2: Strip HTML (if present)
    before_html_strip = len(email_body)
    email_body = strip_html(email_body)
    after_html_strip = len(email_body)
    app.logger.info(f"After strip_html: {before_html_strip} → {after_html_strip} chars")
    
    if after_html_strip < 10:
        app.logger.error(f"❌ CRITICAL: Content too short after HTML stripping!")
        app.logger.error(f"Returning original body to preserve content")
        return original_body.strip()
    
    # Step 3: CRITICAL - Remove quoted email content (our template that we sent)
    # Look for common email reply separators
    quote_markers = [
        r'\n\s*On\s+.+?wrote:',
        r'\n\s*From:\s*.+\n\s*Sent:',
        r'\n\s*-{3,}\s*Original Message\s*-{3,}',
        r'\n\s*-{3,}\s*Forwarded message\s*-{3,}',
        r'\n\s*Begin forwarded message:',
        r'\n\s*On\s+\w+,\s+\w+\s+\d+,\s+\d{4}\s+at\s+\d+:\d+\s+\w+\s+.+?\s+wrote:',  # "On Wed, Feb 11, 2026 at 8:59 AM wrote:"
    ]
    
    # Find where the quoted content starts
    earliest_quote_pos = len(email_body)
    
    for pattern in quote_markers:
        match = re.search(pattern, email_body, re.IGNORECASE | re.MULTILINE)
        if match and match.start() < earliest_quote_pos:
            earliest_quote_pos = match.start()
            app.logger.info(f"✓ Found quote marker at position {match.start()}: {pattern}")
    
    # Extract only content BEFORE the quoted section
    if earliest_quote_pos < len(email_body):
        email_body = email_body[:earliest_quote_pos].strip()
        app.logger.info(f"✓ Removed quoted content, new length: {len(email_body)} chars")
    
    # Step 4: Remove lines that start with quote markers (>, |, etc.)
    lines = email_body.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that are clearly quotes
        if re.match(r'^\s*[>|]+', line):
            continue
        if re.match(r'^\s*&gt;', line):
            continue
        cleaned_lines.append(line)
    
    email_body = '\n'.join(cleaned_lines)
    
    # Step 5: Check for inline format and reformat
    is_inline = bool(re.search(
        r'(user\s*name.*?contact\s*number|location.*?application)', 
        email_body.lower()
    )) and email_body.count('\n') < 5
    
    if is_inline:
        app.logger.info("✓ Detected INLINE format - reformatting with line breaks")
        
        # Add line breaks before each field
        email_body = re.sub(r'(?i)(User Name[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Contact Number[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Location[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Issue Start Date[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Impacted Application Name[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Application Name[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Any Workaround[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Previous Incident[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Error Screenshot[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Mirror ID[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Issue Description[:/\-\s]*)', r'\n\n\1', email_body)
        
        email_body = email_body.lstrip('\n')
        app.logger.info(f"✓ Reformatted inline: now {len(email_body)} chars, {email_body.count(chr(10))} lines")
    
    # Step 6: Remove ONLY email signatures (DO NOT call extract_new_reply_content!)
    final_content = email_body
    
    # Remove common email signatures
    signatures_to_remove = [
        r'\n\s*Sent from my iPhone.*',
        r'\n\s*Sent from my Android.*',
        r'\n\s*Get Outlook for.*',
        r'\n\s*Sent from Mail.*',
        r'\n\s*Sent from Yahoo.*',
        r'\n\s*Sent from Samsung.*',
    ]
    
    for sig_pattern in signatures_to_remove:
        before = len(final_content)
        final_content = re.sub(sig_pattern, '', final_content, flags=re.IGNORECASE)
        after = len(final_content)
        if before != after:
            app.logger.info(f"✓ Removed signature: {before} → {after} chars")
    
    # Step 7: Clean HTML entities
    final_content = re.sub(r'&quot;', '"', final_content)
    final_content = re.sub(r'&#39;', "'", final_content)
    final_content = re.sub(r'&amp;', '&', final_content)
    final_content = re.sub(r'&lt;', '<', final_content)
    final_content = re.sub(r'&gt;', '>', final_content)
    final_content = re.sub(r'&nbsp;', ' ', final_content)
    
    # Step 8: Clean up excessive blank lines only
    final_content = re.sub(r'\n\n\n+', '\n\n', final_content)
    final_content = final_content.strip()
    
    # Step 9: Final validation and logging
    app.logger.info("=" * 70)
    app.logger.info("CLEAN_TEMPLATE_RESPONSE - RESULT")
    app.logger.info(f"Final length: {len(final_content)} characters")
    app.logger.info(f"Final line count: {final_content.count(chr(10))} lines")
    app.logger.info(f"Final content preview (first 500 chars):")
    app.logger.info(final_content[:500])
    app.logger.info("=" * 70)
    
    # CRITICAL: Return content even if short - it's what the user provided
    if len(final_content) < 3:
        app.logger.warning(f"⚠️  Very short content: '{final_content}'")
        # Return original if we somehow stripped everything
        if len(email_body) > 3:
            app.logger.warning("⚠️  Using original email_body instead")
            return email_body.strip()
        return "(Minimal content - please see attachments)"
    
    return final_content
''''
def clean_template_response(email_body: str) -> str:
    """
    Clean template response while preserving ALL content.
    CRITICAL: This function MUST return the full template content for ServiceNow.
    """
    if not email_body:
        app.logger.error("❌ clean_template_response: email_body is EMPTY or None")
        return "(No content provided by user)"
    
    app.logger.info("=" * 70)
    app.logger.info("CLEAN_TEMPLATE_RESPONSE - START")
    app.logger.info(f"Input length: {len(email_body)} characters")
    app.logger.info(f"First 200 chars of input: {email_body[:200]}")
    app.logger.info("=" * 70)
    
    # Step 1: Convert literal \n to actual newlines
    original_body = email_body
    email_body = email_body.replace('\\n', '\n')
    if email_body != original_body:
        app.logger.info(f"✓ Converted literal \\n to newlines")
    
    # Step 2: Strip HTML (if present)
    before_html_strip = len(email_body)
    email_body = strip_html(email_body)
    after_html_strip = len(email_body)
    app.logger.info(f"After strip_html: {before_html_strip} → {after_html_strip} chars")
    
    if after_html_strip < 10:
        app.logger.error(f"❌ CRITICAL: Content too short after HTML stripping!")
        app.logger.error(f"Returning original body to preserve content")
        # Return original if HTML stripping destroyed content
        return original_body.strip()
    
    # Step 3: Check for inline format and reformat
    is_inline = bool(re.search(
        r'(user\s*name.*?contact\s*number|location.*?application)', 
        email_body.lower()
    )) and email_body.count('\n') < 5
    
    if is_inline:
        app.logger.info("✓ Detected INLINE format - reformatting with line breaks")
        
        # Add line breaks before each field
        email_body = re.sub(r'(?i)(User Name[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Contact Number[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Location[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Issue Start Date[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Impacted Application Name[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Application Name[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Any Workaround[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Previous Incident[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Error Screenshot[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Mirror ID[:/\-\s]*)', r'\n\1', email_body)
        email_body = re.sub(r'(?i)(Issue Description[:/\-\s]*)', r'\n\n\1', email_body)
        
        email_body = email_body.lstrip('\n')
        app.logger.info(f"✓ Reformatted inline: now {len(email_body)} chars, {email_body.count(chr(10))} lines")
    
    # Step 4: Remove ONLY email signatures (DO NOT call extract_new_reply_content!)
    final_content = email_body
    
    # Remove common email signatures
    signatures_to_remove = [
        r'\n\s*Sent from my iPhone.*',
        r'\n\s*Sent from my Android.*',
        r'\n\s*Get Outlook for.*',
        r'\n\s*Sent from Mail.*',
        r'\n\s*Sent from Yahoo.*',
        r'\n\s*Sent from Samsung.*',
    ]
    
    for sig_pattern in signatures_to_remove:
        before = len(final_content)
        final_content = re.sub(sig_pattern, '', final_content, flags=re.IGNORECASE)
        after = len(final_content)
        if before != after:
            app.logger.info(f"✓ Removed signature: {before} → {after} chars")
    
    # Step 5: Clean up excessive blank lines only
    final_content = re.sub(r'\n\n\n+', '\n\n', final_content)
    final_content = final_content.strip()
    
    # Step 6: Final validation and logging
    app.logger.info("=" * 70)
    app.logger.info("CLEAN_TEMPLATE_RESPONSE - RESULT")
    app.logger.info(f"Final length: {len(final_content)} characters")
    app.logger.info(f"Final line count: {final_content.count(chr(10))} lines")
    app.logger.info(f"Final content preview (first 300 chars):")
    app.logger.info(final_content[:300])
    app.logger.info("=" * 70)
    
    # CRITICAL: Return content even if short - it's what the user provided
    if len(final_content) < 3:
        app.logger.warning(f"⚠️  Very short content: '{final_content}'")
        # Return original if we somehow stripped everything
        if len(email_body) > 3:
            app.logger.warning("⚠️  Using original email_body instead")
            return email_body.strip()
        return "(Minimal content - please see attachments)"
    
    return final_content
'''

def clean_subject_for_short_description(subject: str) -> str:
    """
    Enhanced cleaning of subject line for use in short description.
    Removes:
    - Re:/Fwd:/FW: prefixes
    - Incident numbers
    - Template-related text
    - "Complete Required Fields" (repeated or single)
    - Multiple consecutive dashes/hyphens
    - Duplicate content
    - Trailing/leading special characters
    """
    if not subject:
        return ""
    
    # Step 1: Remove Re:, Fwd:, FW: prefixes (case-insensitive, multiple occurrences)
    cleaned = re.sub(r'^(?:\s*(re|fwd|fw)\s*[:\-]\s*)+', '', subject, flags=re.I)
    
    # Step 2: Remove "Additional Information Required" and similar template markers
    cleaned = re.sub(r'\s*-?\s*Additional Information Required\s*', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\s*-?\s*Info Required\s*', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\s*-?\s*More Details Needed\s*', '', cleaned, flags=re.I)
    
    # Step 3: Remove "Please Complete Required Fields" (handles multiple occurrences)
    cleaned = re.sub(r'\s*-?\s*Please Complete Required Fields\s*', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\s*-?\s*Complete Required Fields\s*', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\s*-?\s*Required Fields\s*', '', cleaned, flags=re.I)
    
    # Step 4: Remove incident number patterns like -[INC0012345] or -(INC0012345)
    cleaned = re.sub(r'\s*-?\s*[\[\(]INC\d+[\]\)]\s*', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\s*-?\s*INC\d+\s*', '', cleaned, flags=re.I)
    
    # Step 5: Remove multiple consecutive dashes/hyphens (e.g., "- - -" or "---")
    cleaned = re.sub(r'\s*-\s*-\s*-+', ' ', cleaned)  # Multiple spaced dashes
    cleaned = re.sub(r'-{2,}', '-', cleaned)  # Consecutive dashes
    
    # Step 6: Remove standalone dashes at beginning or end
    cleaned = re.sub(r'^[\s\-:]+|[\s\-:]+$', '', cleaned)
    
    # Step 7: Clean up extra spaces around pipes (|)
    cleaned = re.sub(r'\s*\|\s*', ' | ', cleaned)
    
    # Step 8: Remove duplicate content (handles repeated phrases)
    # Split by delimiters and remove duplicates while preserving order
    parts = re.split(r'\s*[\-|]\s*', cleaned)
    seen = set()
    unique_parts = []
    for part in parts:
        part_lower = part.strip().lower()
        if part_lower and part_lower not in seen and len(part_lower) > 2:
            seen.add(part_lower)
            unique_parts.append(part.strip())
    
    # Rejoin with appropriate separator
    if unique_parts:
        cleaned = ' | '.join(unique_parts)
    else:
        cleaned = ''
    
    # Step 9: Final cleanup - collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Step 10: Remove trailing/leading special characters again (after all processing)
    cleaned = re.sub(r'^[\s\-:|]+|[\s\-:|]+$', '', cleaned)
    
    # If we ended up with empty string, return a default
    if not cleaned or len(cleaned) < 3:
        return "Support Request"
    
    return cleaned

REQUIRED_FIELDS = [
    "user_name",
    "contact_number", 
    "location",
    "issue_start_date",
    "application_name",
    "issue_description"
]

FIELD_DISPLAY_NAMES = {
    "user_name": "User Name / ID / Email",
    "contact_number": "Contact Number",
    "location": "Location",
    "issue_start_date": "Issue Start Date",
    "application_name": "Impacted Application Name",
    "workaround": "Any Workaround Available",
    "previous_incident": "Previous Incident Reference",
    "mirror_id": "Mirror ID",
    "issue_description": "Issue Description"
}


def extract_field_value(email_body: str, field_name: str) -> str:
    """
    Extract value for a specific field.
    Uses strict end-of-line patterns to prevent cross-contamination between fields.
    """
    if not email_body:
        return ""

    patterns = {
        "user_name": [
            r'user\s*name\s*/?\s*id\s*/?\s*email\s*[:\-]\s*([^\n]+)',
            r'user\s*name\s*[:\-]\s*([^\n]+)',
            r'user\s*id\s*[:\-]\s*([^\n]+)',
        ],
        "contact_number": [
            r'contact\s*number\s*[:\-]\s*([^\n]+)',
            r'phone\s*[:\-]\s*([^\n]+)',
        ],
        "location": [
            r'location\s*[:\-]\s*([^\n]+)',
        ],
        "issue_start_date": [
            r'issue\s*start\s*date\s*[:\-]\s*([^\n]+)',
            r'start\s*date\s*[:\-]\s*([^\n]+)',
        ],
        "application_name": [
            r'impacted\s*application\s*name\s*[:\-]\s*([^\n]+)',
            r'application\s*name\s*[:\-]\s*([^\n]+)',
        ],
        "workaround": [
            r'any\s*workaround\s*available\s*\??\s*[:\-]?\s*([^\n]+)',
        ],
        "previous_incident": [
            r'previous\s*incident\s*reference[^:\-\n]*[:\-]\s*([^\n]+)',
            r'previous\s*incident\s*[:\-]\s*([^\n]+)',
        ],
        "mirror_id": [
            r'mirror\s*id\s*(?:\([^)]*\))?\s*[:\-]\s*([^\n]+)',
        ],
        "issue_description": [
            # Multi-line: everything after the label until a blank line or end of string
            r'issue\s*description\s*[:\-]?\s*\n([\s\S]+?)(?:\n\s*\n|$)',
            r'issue\s*description\s*[:\-]\s*([^\n]+)',
        ],
    }

    # Keywords that signal a NEW field — used to detect cross-contamination
    FIELD_LABEL_RE = re.compile(
        r'^(?:user\s*name|contact\s*number|location|issue\s*start\s*date|'
        r'impacted\s*application|application\s*name|any\s*workaround|'
        r'previous\s*incident|mirror\s*id|issue\s*description)',
        re.I
    )

    for pattern in patterns.get(field_name, []):
        match = re.search(pattern, email_body, re.IGNORECASE | re.MULTILINE)
        if not match:
            continue

        value = match.group(1).strip()

        # Skip placeholders
        if re.match(r'^\(.*\)$', value) or re.match(r'^\[.*\]$', value):
            continue

        # Skip empty/null indicators
        if value.lower() in ['n/a', 'na', 'none', '-', '', 'nil', 'not applicable']:
            continue

        # Skip values that are actually another field label (cross-contamination guard)
        if FIELD_LABEL_RE.match(value):
            continue

        # Field-specific validation
        if field_name == "issue_description" and len(value) < 10:
            continue
        if field_name == "contact_number":
            if len(re.sub(r'[^\d]', '', value)) < 7:
                continue
        if field_name == "user_name" and len(value) < 3:
            continue

        return value

    return ""



def validate_template_fields(email_body: str) -> tuple:
    """
    Validate if all required fields are filled.
    Returns: (is_valid, missing_fields, extracted_values)
    """
    if not email_body:
        return False, list(FIELD_DISPLAY_NAMES.values()), {}
    
    extracted_values = {}
    missing_fields = []
    
    for field_name in REQUIRED_FIELDS:
        value = extract_field_value(email_body, field_name)
        extracted_values[field_name] = value
        
        if not value:
            display_name = FIELD_DISPLAY_NAMES.get(field_name, field_name)
            missing_fields.append(display_name)
    
    is_valid = len(missing_fields) == 0
    
    return is_valid, missing_fields, extracted_values


def get_incomplete_template_response(missing_fields: list, extracted_values: dict = None) -> str:
    """
    If extracted_values is provided → send pre-filled template (green = filled, yellow = missing).
    If extracted_values is None / empty → send blank template (backward compatible).
    """

    # ── Fallback: no values provided, send original blank template ───────────
    if not extracted_values:
        missing_fields_html = "".join(
            f"<li><strong>{f}</strong></li>" for f in missing_fields
        )
        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body>
  <div style="background-color:#fff3cd;border:1px solid #ffc107;border-radius:8px;padding:20px;margin-bottom:20px;">
    <h3 style="color:#856404;margin-top:0;">⚠️ Incomplete Information</h3>
    <p>Thank you for your response. However, we need <strong>all required fields</strong> to be filled to create your incident.</p>
    <p><strong>The following fields are missing or incomplete:</strong></p>
    <ul style="color:#856404;margin:10px 0;">{missing_fields_html}</ul>
  </div>
  <hr style="border:none;border-top:2px solid #e0e0e0;margin:25px 0;">
  <div style="margin-top:20px;">
    <h3 style="color:#2563eb;">📋 Please Provide Complete Information</h3>
    <p><strong>Reply to this email with ALL the required information below:</strong></p>
    <table border="1" cellpadding="10" cellspacing="0" style="border-collapse:collapse;width:100%;max-width:800px;margin:15px 0;">
      <tr><td style="background-color:#f0f0f0;font-weight:bold;width:45%;">User Name / ID / Email: <span style="color:red;">*</span></td><td style="width:55%;background-color:#ffffff;"></td></tr>
      <tr><td style="background-color:#f0f0f0;font-weight:bold;">Contact Number: <span style="color:red;">*</span></td><td style="background-color:#ffffff;"></td></tr>
      <tr><td style="background-color:#f0f0f0;font-weight:bold;">Location: <span style="color:red;">*</span></td><td style="background-color:#ffffff;color:#666;">(Terminal / Building / Landside / Airside / Desk / Office etc.)</td></tr>
      <tr><td style="background-color:#f0f0f0;font-weight:bold;">Issue Start Date: <span style="color:red;">*</span></td><td style="background-color:#ffffff;"></td></tr>
      <tr><td style="background-color:#f0f0f0;font-weight:bold;">Impacted Application Name: <span style="color:red;">*</span></td><td style="background-color:#ffffff;"></td></tr>
      <tr><td style="background-color:#f0f0f0;font-weight:bold;">Any Workaround Available?</td><td style="background-color:#ffffff;"></td></tr>
      <tr><td style="background-color:#f0f0f0;font-weight:bold;">Previous Incident Reference (if any):</td><td style="background-color:#ffffff;"></td></tr>
      <tr><td style="background-color:#f0f0f0;font-weight:bold;">Error Screenshot:</td><td style="background-color:#ffffff;color:#666;">(Please attach)</td></tr>
      <tr><td style="background-color:#f0f0f0;font-weight:bold;">Mirror ID (if access-related):</td><td style="background-color:#ffffff;"></td></tr>
    </table>
    <h3 style="color:#2563eb;margin-top:25px;">Issue Description: <span style="color:red;">*</span></h3>
    <table border="1" cellpadding="10" cellspacing="0" style="border-collapse:collapse;width:100%;max-width:800px;margin:15px 0;">
      <tr><td style="min-height:80px;background-color:#ffffff;color:#666;">[Please describe your issue in detail here]</td></tr>
    </table>
    <p style="color:red;"><strong>* = Required field</strong></p>
    <hr style="border:none;border-top:2px solid #e0e0e0;margin:25px 0;">
    <p>Once you provide <strong>all the required information</strong>, we will create your incident immediately.</p>
    <p>Thank you,<br/><strong>Support Team</strong></p>
  </div>
</body></html>"""

    # ── Main path: pre-fill with whatever the user already provided ──────────
    FIELD_DISPLAY_NAMES = {
        "user_name":         "User Name / ID / Email",
        "contact_number":    "Contact Number",
        "location":          "Location",
        "issue_start_date":  "Issue Start Date",
        "application_name":  "Impacted Application Name",
        "workaround":        "Any Workaround Available",
        "previous_incident": "Previous Incident Reference",
        "mirror_id":         "Mirror ID",
        "issue_description": "Issue Description"
    }
    REQUIRED_FIELDS = [
        "user_name", "contact_number", "location",
        "issue_start_date", "application_name", "issue_description"
    ]

    missing_set = set(missing_fields)

    def field_row(field_name: str, placeholder: str = "") -> str:
        display = FIELD_DISPLAY_NAMES.get(field_name, field_name)
        value   = (extracted_values.get(field_name) or "").strip()
        star    = '<span style="color:red;">*</span>' if field_name in REQUIRED_FIELDS else ''

        if value:
            return (
                f'<tr>'
                f'<td style="background-color:#d4edda;font-weight:bold;width:45%;padding:10px;border:1px solid #c3e6cb;">'
                f'✅ {display}: {star}</td>'
                f'<td style="background-color:#d4edda;width:55%;padding:10px;border:1px solid #c3e6cb;'
                f'color:#155724;font-weight:bold;">{value}</td>'
                f'</tr>'
            )
        else:
            hint = (f'<span style="color:#999;font-size:12px;">{placeholder}</span>'
                    if placeholder else '')
            return (
                f'<tr>'
                f'<td style="background-color:#fff3cd;font-weight:bold;width:45%;padding:10px;border:2px solid #ffc107;">'
                f'⚠️ {display}: {star}</td>'
                f'<td style="background-color:#fff3cd;width:55%;padding:10px;border:2px solid #ffc107;">'
                f'{hint}</td>'
                f'</tr>'
            )

    def description_row() -> str:
        value = (extracted_values.get("issue_description") or "").strip()
        if value:
            safe = value.replace('\n', '<br/>')
            return (
                f'<tr><td colspan="2" style="background-color:#d4edda;padding:10px;'
                f'border:1px solid #c3e6cb;color:#155724;">'
                f'<strong>✅ Issue Description:</strong><br/>{safe}</td></tr>'
            )
        else:
            return (
                f'<tr><td colspan="2" style="background-color:#fff3cd;padding:10px;'
                f'border:2px solid #ffc107;">'
                # f'<strong>⚠️ Issue Description: <span style="color:red;">*</span></strong><br/>'
                f'<span style="color:#999;font-size:12px;">'
                f'[Please describe your issue in detail here]</span></td></tr>'
            )

    mc = len(missing_fields)
    missing_list_html = "".join(f"<li><strong>{f}</strong></li>" for f in missing_fields)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;color:#333;margin:20px;">

  <div style="background-color:#fff3cd;border:2px solid #ffc107;border-radius:8px;padding:20px;margin-bottom:20px;">
    <h3 style="color:#856404;margin-top:0;">
      ⚠️ Almost There! {mc} Required Field{"s" if mc > 1 else ""} Still Missing
    </h3>
    <p>Thank you for your response. We have saved the information you provided.</p>
    <p><strong>Please fill in the highlighted field{"s" if mc > 1 else ""} below and reply:</strong></p>
    <ul style="color:#856404;margin:5px 0 0 0;">{missing_list_html}</ul>
  </div>

  <table style="margin-bottom:15px;font-size:13px;">
    <tr>
      <td style="background-color:#d4edda;border:1px solid #c3e6cb;padding:6px 14px;color:#155724;border-radius:4px;">
        ✅ Already filled — no action needed
      </td>
      <td style="width:20px;"></td>
      <td style="background-color:#fff3cd;border:2px solid #ffc107;padding:6px 14px;color:#856404;border-radius:4px;">
        ⚠️ Missing — please fill this in
      </td>
    </tr>
  </table>

  <h3 style="color:#2563eb;">📋 Your Information (reply with ⚠️ fields completed)</h3>

  <table cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;width:100%;max-width:800px;margin:10px 0;">
    {field_row("user_name")}
    {field_row("contact_number")}
    {field_row("location", "Terminal / Building / Landside / Airside / Desk / Office etc.")}
    {field_row("issue_start_date")}
    {field_row("application_name")}
    {field_row("workaround")}
    {field_row("previous_incident")}
    <tr>
      <td style="background-color:#f0f0f0;font-weight:bold;padding:10px;border:1px solid #ccc;">
        Error Screenshot:
      </td>
      <td style="background-color:#ffffff;padding:10px;border:1px solid #ccc;color:#666;font-size:12px;">
        (Please attach if applicable)
      </td>
    </tr>
    {field_row("mirror_id")}
  </table>

  <h3 style="color:#2563eb;margin-top:20px;">Issue Description: <span style="color:red;">*</span></h3>
  <table cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;width:100%;max-width:800px;margin:10px 0;">
    {description_row()}
  </table>

  <p style="color:red;font-size:13px;"><strong>* = Required field</strong></p>

  <hr style="border:none;border-top:2px solid #e0e0e0;margin:25px 0;">

  <p>
    Simply <strong>reply to this email</strong> with the ⚠️ highlighted fields filled in.<br/>
    Your existing information will be carried over automatically.<br/>
    Once all required fields are provided, we will create your incident immediately.
  </p>

  <p>Thank you,<br/><strong>Support Team</strong></p>
</body>
</html>"""


def build_enhanced_description_from_fields(extracted_values: dict) -> str:
    """Build well-formatted description from extracted values."""
    lines = []
    
    field_order = [
        "user_name", "contact_number", "location", "issue_start_date",
        "application_name", "workaround", "previous_incident", "mirror_id",
        "issue_description"
    ]
    
    for field_name in field_order:
        value = extracted_values.get(field_name, "")
        if value:
            display_name = FIELD_DISPLAY_NAMES.get(field_name, field_name)
            
            if field_name == "issue_description":
                lines.append(f"\n{display_name}:")
                lines.append(value)
            else:
                lines.append(f"{display_name}: {value}")
    
    return "\n".join(lines)



def get_last_public_comment(sys_id: str) -> Optional[dict]:
    """Return last Additional comments entry for an incident."""
    base_url = f"https://{SN_INSTANCE}.service-now.com"
    url = f"{base_url}/api/now/v1/table/sys_journal_field"
    params = {
        "sysparm_query": f"element_id={sys_id}^element=comments^ORDERBYDESCsys_created_on",
        "sysparm_limit": "1",
        "sysparm_fields": "value,sys_created_on,sys_created_by",
        "sysparm_display_value": "true",
        "sysparm_exclude_reference_link": "true",
    }
    try:
        resp = requests.get(url, params=params, auth=(SN_USER, SN_PASSWORD), timeout=30)
        resp.raise_for_status()
        items = resp.json().get("result", []) or []
        return items[0] if items else None
    except Exception as e:
        app.logger.warning(f"Failed to fetch last comment for {sys_id}: {e}")
        return None

def verify_incident_caller(incident_number: str, expected_caller_email: str) -> bool:
    """
    Verify that the incident has the correct caller set.
    Returns True if caller is correctly set, False otherwise.
    """
    try:
        incident = get_servicenow_incident(number=incident_number)
        if not incident:
            app.logger.error(f"Could not fetch incident {incident_number} for verification")
            return False
        
        caller_id = incident.get("caller_id")
        
        if not caller_id:
            app.logger.error(f"✗ Incident {incident_number} has NO caller_id set!")
            return False
        
        # If caller_id is a dict (display_value format), extract the value
        if isinstance(caller_id, dict):
            caller_display = caller_id.get("display_value", "")
            app.logger.info(f"✓ Incident {incident_number} has caller: {caller_display}")
        else:
            app.logger.info(f"✓ Incident {incident_number} has caller_id: {caller_id}")
        
        return True
        
    except Exception as e:
        app.logger.error(f"Error verifying caller for incident {incident_number}: {e}")
        return False

# ---------------- PROCESS EMAILS WITH TEMPLATE FLOW ---------------- #

def process_unread_emails(service):
    """
    Enhanced email processing:
    - CC skip addresses receive template request instead of being skipped
    - When they reply with filled template, incident is created with all functionalities
    - Closed incident handling: sends status + template, creates new incident on reply
    """
    results = []
    blocked_count = 0
    besafe_count = 0
    next_token = None

    while True:
        list_kwargs = {"userId": "me", "labelIds": ["UNREAD", "INBOX"], "maxResults": 50}
        if next_token:
            list_kwargs["pageToken"] = next_token

        res = service.users().messages().list(**list_kwargs).execute()
        messages = res.get("messages", [])
        next_token = res.get("nextPageToken")

        if not messages:
            break

        for ref in messages:
            try:
                msg = service.users().messages().get(
                    userId="me", id=ref["id"], format="full"
                ).execute()

                headers = msg.get("payload", {}).get("headers", [])
                sender = get_header(headers, "From", default="unknown@example.com")
                subject = get_header(headers, "Subject", default="(No Subject)")
                body = extract_body(msg.get("payload", {}))
                thread_id = msg.get("threadId")

                # ============ CHECK CC SKIP ADDRESSES - NEW BEHAVIOR ============
                cc_addresses = get_cc_addresses(headers)
                
                print("\n" + "="*70)
                print("PROCESSING EMAIL")
                print("From   :", sender)
                print("Subject:", subject)
                print("Thread :", thread_id)
                print("CC addresses:", cc_addresses)
                print("Skip list:", CC_SKIP_ADDRESSES)

                app.logger.info(f"=" * 60)
                app.logger.info(f"Processing email from: {sender}")
                app.logger.info(f"Subject: {subject}")
                app.logger.info(f"Thread ID: {thread_id}")
                app.logger.info(f"CC addresses found: {cc_addresses}")

                # Check if we should handle this email specially due to CC skip addresses
                has_skip_address = False
                skip_reason = None
                if cc_addresses and CC_SKIP_ADDRESSES:
                    cc_normalized = {addr.lower().strip() for addr in cc_addresses}
                    skip_set = {addr.lower().strip() for addr in CC_SKIP_ADDRESSES}

                    for skip_addr in skip_set:
                        if skip_addr in cc_normalized:
                            has_skip_address = True
                            skip_reason = skip_addr
                            break

                # ============ NEW LOGIC FOR CC SKIP ADDRESSES ============
                if has_skip_address:
                    app.logger.info(f"✓ Email has CC skip address: {skip_reason}")
                    
                    # FIRST TIME EMAIL WITH CC SKIP - SEND BA IT SERVICE DESK TEMPLATE
                    app.logger.info(f"Sending BA IT Service Desk template to CC skip address email: {sender}")
                        
                    try:
                        # Use BA IT Service Desk specific template for CC skip addresses
                        template = get_cc_skip_template()
                        template_subject = f"Re: {subject}"
                            
                        # Send as plain text (not HTML) for the BA template
                        send_email(service, sender, template_subject, template, is_html=False)
                            
                        # Mark as template pending to track if they reply with filled info
                        if thread_id:
                            _mark_template_pending(thread_id, sender, subject)
                            
                        app.logger.info(f"✓ BA IT Service Desk template sent to CC skip address: {sender}")
                            
                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "cc": cc_addresses,
                            "decision": {"action": "CC_SKIP_BA_TEMPLATE_SENT"},
                            "result": {
                                "type": "CC_SKIP_BA_TEMPLATE_SENT",
                                "cc_skip_address": skip_reason,
                                "template_pending": True,
                                "thread_id": thread_id,
                                "message": "BA IT Service Desk auto-response sent"
                            },
                            "response_mail": template
                        }
                            
                    except Exception as e:
                        app.logger.error(f"Failed to send BA template to CC skip address: {e}")
                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "cc": cc_addresses,
                            "decision": {"action": "CC_SKIP_TEMPLATE_FAILED"},
                            "error": str(e),
                            "result": {"type": "CC_SKIP_TEMPLATE_FAILED"},
                        }

                    # Mark as read
                    try:
                        service.users().messages().modify(
                            userId="me",
                            id=msg["id"],
                            body={"removeLabelIds": ["UNREAD"]}
                        ).execute()
                    except Exception as mark_err:
                        outcome["mark_read_error"] = str(mark_err)
                    
                    results.append(outcome)
                    app.logger.info(f"=" * 60)
                    continue  # ← Move to next email

                # If we reach here, no CC skip address - proceed with normal flow
                app.logger.info(f"No CC skip address - proceeding with normal processing")
                app.logger.info(f"=" * 60)
                
                # ============ STEP 0: CHECK FOR BE SAFE PASSWORD RESET (HIGHEST PRIORITY) ============
                if is_besafe_password_reset(subject, body):
                    besafe_count += 1
                    app.logger.info(f"✓ BE SAFE password reset detected from: {sender}")
                    
                    try:
                        # Send BE SAFE template
                        besafe_template = get_besafe_password_reset_template()
                        reply_subject = f"Re: {subject}"
                        
                        send_email(service, sender, reply_subject, besafe_template, is_html=False)
                        
                        app.logger.info(f"✓ BE SAFE password reset template sent to: {sender}")
                        
                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "decision": {"action": "BESAFE_PASSWORD_RESET"},
                            "result": {
                                "type": "BESAFE_PASSWORD_RESET_TEMPLATE_SENT",
                                "no_incident_created": True,
                                "message": "BE SAFE password reset template sent - no incident created"
                            },
                            "response_mail": besafe_template
                        }
                        
                    except Exception as e:
                        app.logger.error(f"Failed to send BE SAFE template: {e}")
                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "decision": {"action": "BESAFE_TEMPLATE_FAILED"},
                            "error": str(e),
                            "result": {"type": "BESAFE_TEMPLATE_FAILED"},
                        }
                    
                    # ✓ OPTIONAL: Only mark as read if you want them to "disappear"
                    # RECOMMENDED: Comment this out to keep emails visible in inbox
                    
                    try:
                        service.users().messages().modify(
                            userId="me",
                            id=msg["id"],
                            body={"removeLabelIds": ["UNREAD"]}
                        ).execute()
                    except Exception as mark_err:
                        outcome["mark_read_error"] = str(mark_err)
                    
                    
                    results.append(outcome)  # ✓ ADD THIS: Show in dashboard
                    app.logger.info(f"=" * 60)
                    continue
                
                # ============ STEP 1: CHECK IF BLOCKED DOMAIN ============
                if is_blocked_sender(sender):
                    blocked_count += 1
                    app.logger.info(f"Blocked domain detected: {sender}. Checking similarity first.")
                    
                    # First, check ML similarity
                    decision = generate_response(body, threshold=0.60)
                    
                    if decision["action"] == "AUTO_RESPONSE":
                        # Similarity match found - send auto-response
                        app.logger.info(f"Similarity match for blocked domain: {sender} (score: {decision.get('similarity', 0):.3f})")
                        
                        try:
                            reply_text = decision["response"]
                            send_email(service, sender, f"Re: {subject}", reply_text)
                            
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": decision,
                                "result": {
                                    "type": "BLOCKED_DOMAIN_AUTO_RESPONSE",
                                    "blocked_domain": _extract_domain(sender),
                                    "similarity": decision.get("similarity", 0)
                                },
                                "response_mail": reply_text
                            }
                            app.logger.info(f"Successfully sent auto-response to blocked domain: {sender}")
                            
                        except Exception as e:
                            app.logger.error(f"Failed to send auto-response to blocked domain {sender}: {e}")
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": decision,
                                "error": str(e),
                                "result": {"type": "BLOCKED_DOMAIN_RESPONSE_FAILED"},
                            }
                    
                    else:
                        # No similarity match - send self-service template
                        app.logger.info(f"No similarity match for blocked domain: {sender}. Sending self-service template.")
                        
                        try:
                            template = get_blocked_domain_template()
                            template_subject = f"Re: {subject} - Self-Service Portal Information"
                            send_email(service, sender, template_subject, template)
                            
                            app.logger.info(f"Successfully sent self-service template to blocked domain: {sender}")
                            
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": {"action": "BLOCKED_DOMAIN_TEMPLATE_SENT"},
                                "result": {
                                    "type": "BLOCKED_DOMAIN_SELF_SERVICE",
                                    "blocked_domain": _extract_domain(sender)
                                },
                                "response_mail": template
                            }
                            
                        except Exception as e:
                            app.logger.error(f"Failed to send template to blocked domain {sender}: {e}")
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": {"action": "BLOCKED_DOMAIN_TEMPLATE_FAILED"},
                                "error": str(e),
                                "result": {"type": "BLOCKED_DOMAIN_TEMPLATE_FAILED"},
                            }
                    
                    # Mark as read (based on BLOCKED_MARK_READ config)
                    if BLOCKED_MARK_READ:
                        try:
                            service.users().messages().modify(
                                userId="me",
                                id=msg["id"],
                                body={"removeLabelIds": ["UNREAD"]}
                            ).execute()
                        except Exception as mark_err:
                            outcome["mark_read_error"] = str(mark_err)
                    
                    results.append(outcome)
                    continue
                
                # ============ STEP 2: NOT BLOCKED DOMAIN - NORMAL FLOW ============

                if "additional information required" in (subject or "").lower():
                    app.logger.info("✓✓✓ SUBJECT CONTAINS 'Additional Information Required' - FORCING TEMPLATE RESPONSE MODE")
                    is_template_reply = True
                else:
                    # Fallback to other detection methods
                    is_template_reply = _is_template_pending(thread_id) or is_template_response(body, subject)

                # Enhanced debug logging
                app.logger.info(f"")
                app.logger.info(f"═══════════════════════════════════════════════════════")
                app.logger.info(f"TEMPLATE DETECTION DEBUG")
                app.logger.info(f"From: {sender}")
                app.logger.info(f"Subject: {subject}")
                app.logger.info(f"Thread ID: {thread_id}")
                app.logger.info(f"Template Pending (by thread): {_is_template_pending(thread_id)}")
                app.logger.info(f"Subject has 'Additional Info Required': {'additional information required' in (subject or '').lower()}")
                app.logger.info(f"is_template_response(body, subject): {is_template_response(body, subject)}")
                app.logger.info(f"FINAL DECISION: is_template_reply = {is_template_reply}")
                app.logger.info(f"Email Body Preview (first 500 chars):")
                app.logger.info(f"{body[:500] if body else 'EMPTY'}")
                app.logger.info(f"═══════════════════════════════════════════════════════")
                app.logger.info(f"")
                
                if is_template_reply:
                    app.logger.info(f"Processing template response from {sender}")
                    app.logger.info(f"Thread ID: {thread_id}, Template Pending: {_is_template_pending(thread_id)}")
                    
                    # Clean the template response
                    if not body or len(body.strip()) < 5:
                        app.logger.error("❌ CRITICAL: Email body is EMPTY or too short!")
                        cleaned_template = "User replied with attachments - see attached files for details"
                        is_valid = False
                        missing_fields = REQUIRED_FIELDS
                        extracted_values = {}
                    else:
                        cleaned_template = clean_template_response(body)
                        if not cleaned_template or len(cleaned_template) < 10:
                            app.logger.warning(f"Cleaned template is empty or too short: '{cleaned_template}'")
                            cleaned_template = body if body and len(body) > 10 else "User replied to template (content may be in attachments)"
                        
                        # ═══════════════════════════════════════════════════════════
                        # NEW: VALIDATE TEMPLATE FIELDS BEFORE CREATING INCIDENT
                        # ═══════════════════════════════════════════════════════════
                        app.logger.info("=" * 70)
                        app.logger.info("TEMPLATE VALIDATION - START")
                        
                        is_valid, missing_fields, extracted_values = validate_template_fields(cleaned_template)
                        
                        app.logger.info(f"Template Valid: {is_valid}")
                        app.logger.info(f"Missing Fields: {missing_fields}")
                        app.logger.info(f"Extracted Values: {extracted_values}")
                        app.logger.info("=" * 70)
                    
                    # ═══════════════════════════════════════════════════════════
                    # DECISION: Create incident OR request complete information
                    # ═══════════════════════════════════════════════════════════
                    
                    if not is_valid:
                        # Template is INCOMPLETE - send request for missing fields
                        app.logger.info(f"⚠️  Template incomplete. Missing {len(missing_fields)} required fields")
                        app.logger.info(f"Missing fields: {', '.join(missing_fields)}")
                        
                        try:
                            incomplete_response = get_incomplete_template_response(missing_fields, extracted_values)
                            reply_subject = f"Re: {subject} - Please Complete Required Fields"
                            
                            send_email(service, sender, reply_subject, incomplete_response, is_html=True)
                            
                            app.logger.info(f"✓ Incomplete template response sent to {sender}")
                            app.logger.info(f"  Thread remains in TEMPLATE_PENDING state")
                            
                            # Keep template pending - don't clear it
                            # User needs to reply with complete information
                            
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": {"action": "TEMPLATE_INCOMPLETE"},
                                "result": {
                                    "type": "TEMPLATE_VALIDATION_FAILED",
                                    "missing_fields": missing_fields,
                                    "missing_count": len(missing_fields),
                                    "thread_id": thread_id,
                                    "template_still_pending": True
                                },
                                "response_mail": incomplete_response
                            }
                            
                        except Exception as e:
                            app.logger.error(f"Failed to send incomplete template response: {e}")
                            fallback = f"Please provide all required information to create your incident. Missing fields: {', '.join(missing_fields)}"
                            try:
                                send_email(service, sender, f"Re: {subject}", fallback)
                            except:
                                pass
                            
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": {"action": "TEMPLATE_INCOMPLETE_SEND_FAILED"},
                                "error": str(e),
                                "result": {"type": "TEMPLATE_INCOMPLETE_RESPONSE_FAILED"},
                            }
                        
                        # Mark as read
                        try:
                            service.users().messages().modify(
                                userId="me",
                                id=msg["id"],
                                body={"removeLabelIds": ["UNREAD"]}
                            ).execute()
                            app.logger.info(f"✓ Email marked as read")
                        except Exception as mark_err:
                            app.logger.error(f"Failed to mark email as read: {mark_err}")
                            outcome["mark_read_error"] = str(mark_err)
                        
                        results.append(outcome)
                        continue  # ← CRITICAL: Skip to next email - don't create incident yet
                    
                    # ═══════════════════════════════════════════════════════════
                    # Template is COMPLETE - proceed with incident creation
                    # ═══════════════════════════════════════════════════════════
                    app.logger.info("✓✓✓ Template is COMPLETE - creating incident")
                    app.logger.info(f"All {len(REQUIRED_FIELDS)} required fields filled")
                    
                    # Build enhanced description from validated fields
                    description_from_fields = build_enhanced_description_from_fields(extracted_values)
                    
                    # Extract template fields for short description enhancement
                    template_fields = extract_template_fields(cleaned_template)
                    
                    try:
                        app.logger.info(f"Creating incident for complete template response from {sender}")
                        
                        # Log what we're sending
                        app.logger.info("🔍" * 30)
                        app.logger.info("SERVICENOW INCIDENT CREATION - VALIDATED TEMPLATE")
                        app.logger.info(f"Subject: {subject}")
                        app.logger.info(f"Description length: {len(description_from_fields)} characters")
                        app.logger.info(f"Description content:")
                        app.logger.info(description_from_fields)
                        app.logger.info("🔍" * 30)
                        
                        incident = create_servicenow_incident(
                            subject=subject,
                            body=description_from_fields,  # ← Use the formatted description
                            priority="5",
                            correlation_id=thread_id,
                            channel="email",
                            assignment_group="SVC DESC",
                            caller_email=sender,
                            template_fields=template_fields
                        )
                        
                        incident_number = incident.get('number')
                        sys_id = incident.get('sys_id')
                        
                        app.logger.info(f"✓ Incident {incident_number} created successfully from validated template")
                        
                        # Process attachments
                        attachment_result = process_and_upload_attachments(
                            service, 
                            msg["id"], 
                            msg.get("payload", {}), 
                            sys_id
                        )
                        
                        if attachment_result['count'] > 0:
                            app.logger.info(f"✓ Attachments: {attachment_result['successful']}/{attachment_result['count']} uploaded successfully")
                        
                        verify_incident_caller(incident_number, sender)
                        _cache_incident(sender, subject, thread_id, incident)
                        _clear_template_pending(thread_id)  # ← Clear pending status
                        
                        # Build confirmation with field summary
                        field_summary_lines = []
                        for field_name in REQUIRED_FIELDS:
                            value = extracted_values.get(field_name, "")
                            if value:
                                display_name = FIELD_DISPLAY_NAMES.get(field_name, field_name)
                                preview = value[:50] + "..." if len(value) > 50 else value
                                field_summary_lines.append(f"  • {display_name}: {preview}")
                        
                        field_summary = "\n".join(field_summary_lines) if field_summary_lines else "All required information captured"
                        
                        confirmation = f'''
        Thank you! Your request has been logged as incident {incident_number}.

        All required information has been captured


                Incident Details:
                - Number: {incident_number}
                - Priority: 5
                - Channel: Email
                - Assignment group: SVC DESC

        Our support team will review your request and contact you shortly.

        Thank you,
        Support Team'''
                        
                        reply_subject = format_new_incident_subject(incident_number, subject)
                        send_email(service, sender, reply_subject, confirmation)
                        
                        app.logger.info(f"✓ Confirmation email sent to {sender}")
                        
                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "decision": {"action": "INCIDENT_CREATED_FROM_VALIDATED_TEMPLATE"},
                            "result": {
                                "type": "INCIDENT_CREATED_FROM_COMPLETE_TEMPLATE",
                                "incident_number": incident_number,
                                "sys_id": sys_id,
                                "priority": "5",
                                "validated_fields": list(extracted_values.keys()),
                                "location": template_fields.get("location"),
                                "application_name": template_fields.get("application_name"),
                                "attachments": {
                                    "total": attachment_result.get('count', 0),
                                    "successful": attachment_result.get('successful', 0),
                                    "failed": attachment_result.get('failed', 0)
                                }
                            },
                            "response_mail": confirmation
                        }
                        
                    except Exception as e:
                        app.logger.error(f"Failed to create incident from validated template: {e}")
                        app.logger.exception("Full exception details:")
                        
                        fallback = f'''We encountered an issue creating your incident.

                Error details have been logged. Please contact support directly or try again later.

                Error: {str(e)[:100]}

                Thank you,
                Support Team'''
                        
                        try:
                            send_email(service, sender, f"Re: {subject}", fallback)
                        except Exception as email_err:
                            app.logger.error(f"Failed to send error email: {email_err}")
                        
                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "decision": {"action": "VALIDATED_TEMPLATE_INCIDENT_FAILED"},
                            "error": str(e),
                            "result": {"type": "VALIDATED_TEMPLATE_INCIDENT_CREATION_FAILED"},
                        }
                    
                    try:
                        service.users().messages().modify(
                            userId="me",
                            id=msg["id"],
                            body={"removeLabelIds": ["UNREAD"]}
                        ).execute()
                        app.logger.info(f"✓ Email marked as read")
                    except Exception as mark_err:
                        app.logger.error(f"Failed to mark email as read: {mark_err}")
                        outcome["mark_read_error"] = str(mark_err)
                    
                    results.append(outcome)
                    continue  # ← CRITICAL: Skip to next email
                
                # ============ STEP 3: CHECK FOR EXISTING INCIDENT ============
                inc_from_subject = extract_inc_number_from_subject(subject)
                existing_incident = None
                
                if inc_from_subject:
                    try:
                        existing_incident = get_servicenow_incident(number=inc_from_subject)
                    except Exception as e:
                        app.logger.warning(f"SN lookup by number failed: {e}")
                
                existing_cached = _lookup_cached_incident(sender, subject, thread_id)
                
                if not existing_incident and thread_id:
                    try:
                        existing_incident = get_servicenow_incident(correlation_id=thread_id)
                    except Exception as e:
                        app.logger.warning(f"SN lookup by correlation_id failed: {e}")
                
                if not existing_incident and existing_cached:
                    try:
                        existing_incident = get_servicenow_incident(sys_id=existing_cached.get("sys_id"))
                    except Exception as e:
                        app.logger.warning(f"SN lookup by cached sys_id failed: {e}")
                
                # ============ STEP 4: HANDLE EXISTING INCIDENT ============
                if existing_incident:
                    sys_id = existing_incident.get("sys_id")
                    number = existing_incident.get("number")
                    current_state = existing_incident.get("state", "Unknown")
                    is_closed = str(current_state).lower() in ["closed", "resolved", "7", "6"]
                    
                    app.logger.info(f"Processing existing incident {number}, state: {current_state}, is_closed: {is_closed}")
                    
                    # ============ NEW: CHECK IF REPLYING TO CLOSED INCIDENT TEMPLATE ============
                    if _is_closed_incident_pending(thread_id):
                        closed_info = _get_closed_incident_info(thread_id)
                        old_incident_number = closed_info.get("old_incident_number")
                        
                        # User replied with template - create NEW incident
                        app.logger.info(f"User replied to closed incident {old_incident_number} template. Creating NEW incident...")
                        
                        cleaned_template = clean_template_response(body)
                        if not cleaned_template:
                            cleaned_template = body
                        
                        template_fields = extract_template_fields(cleaned_template)
                        
                        try:
                            # Create NEW incident
                            new_incident = create_servicenow_incident(
                                subject=subject,
                                body=cleaned_template,
                                priority="5",
                                correlation_id=thread_id,
                                channel="email",
                                assignment_group="SVC DESC",
                                caller_email=sender,
                                template_fields=template_fields
                            )
                            
                            new_incident_number = new_incident.get('number')
                            new_sys_id = new_incident.get('sys_id')
                            
                            # Process attachments
                            attachment_result = process_and_upload_attachments(
                                service,
                                msg["id"],
                                msg.get("payload", {}),
                                new_sys_id
                            )
                            
                            app.logger.info(f"✓ NEW Incident {new_incident_number} created after closed incident {old_incident_number}")
                            if attachment_result['count'] > 0:
                                app.logger.info(f"✓ Attachments: {attachment_result['successful']}/{attachment_result['count']} uploaded")
                            
                            verify_incident_caller(new_incident_number, sender)
                            _cache_incident(sender, subject, thread_id, new_incident)
                            _clear_closed_incident_pending(thread_id)
                            
                            confirmation = f"""
Thank you! A new incident has been created: {new_incident_number}

Previous incident {old_incident_number} was closed. Your new request has been logged with the information you provided.

New Incident Details:
- Number: {new_incident_number}
- Priority: 5
- Channel: Email
- Assignment group: SVC DESC

Our support team will review your request and contact you shortly.

Thank you,
Support Team"""
                            
                            reply_subject = format_new_incident_subject(new_incident_number, subject)
                            send_email(service, sender, reply_subject, confirmation)
                            
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": {"action": "NEW_INCIDENT_AFTER_CLOSED"},
                                "result": {
                                    "type": "NEW_INCIDENT_CREATED_AFTER_CLOSED",
                                    "old_incident_number": old_incident_number,
                                    "new_incident_number": new_incident_number,
                                    "new_sys_id": new_incident.get("sys_id"),
                                    "location": template_fields.get("location"),
                                    "application_name": template_fields.get("application_name")
                                },
                                "response_mail": confirmation
                            }
                            
                        except Exception as e:
                            app.logger.error(f"Failed to create new incident after closed: {e}")
                            fallback = "We encountered an issue creating your new incident. Please try again."
                            send_email(service, sender, f"Re: {subject}", fallback)
                            
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": {"action": "NEW_INCIDENT_FAILED"},
                                "error": str(e),
                                "result": {"type": "NEW_INCIDENT_CREATION_FAILED"},
                            }
                        
                        try:
                            service.users().messages().modify(
                                userId="me",
                                id=msg["id"],
                                body={"removeLabelIds": ["UNREAD"]}
                            ).execute()
                        except Exception as mark_err:
                            outcome["mark_read_error"] = str(mark_err)
                        
                        results.append(outcome)
                        continue
                    
                    # ============ NEW: IF INCIDENT IS CLOSED, SEND STATUS + TEMPLATE ============
                    if is_closed:
                        app.logger.info(f"Incident {number} is CLOSED. Sending status + template request...")
                        
                        try:
                            # Send combined status + template
                            combined_message = format_closed_incident_status_with_template(existing_incident)
                            reply_subject = f"Re: {subject} - Closed Incident {number}"
                            send_email(service, sender, reply_subject, combined_message, is_html=True)
                            
                            # Mark as pending for new incident creation
                            if thread_id:
                                _mark_closed_incident_pending(thread_id, number, sender, subject)
                            
                            app.logger.info(f"✓ Closed incident status + template sent for {number}")
                            
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": {"action": "CLOSED_INCIDENT_TEMPLATE_SENT"},
                                "result": {
                                    "type": "CLOSED_INCIDENT_STATUS_AND_TEMPLATE",
                                    "incident_number": number,
                                    "sys_id": sys_id,
                                    "state": current_state,
                                    "template_pending": True,
                                    "thread_id": thread_id
                                },
                                "response_mail": combined_message
                            }
                            
                        except Exception as e:
                            app.logger.error(f"Failed to send closed incident template: {e}")
                            fallback = f"Your referenced incident {number} is closed. Please provide details for a new incident."
                            send_email(service, sender, f"Re: {subject}", fallback)
                            
                            outcome = {
                                "from": sender,
                                "subject": subject,
                                "body": body,
                                "decision": {"action": "CLOSED_INCIDENT_TEMPLATE_FAILED"},
                                "error": str(e),
                                "result": {"type": "CLOSED_INCIDENT_TEMPLATE_FAILED"},
                            }
                        
                        try:
                            service.users().messages().modify(
                                userId="me",
                                id=msg["id"],
                                body={"removeLabelIds": ["UNREAD"]}
                            ).execute()
                        except Exception as mark_err:
                            outcome["mark_read_error"] = str(mark_err)
                        
                        results.append(outcome)
                        continue
                    
                    # ============ EXISTING: INCIDENT NOT CLOSED - NORMAL UPDATE FLOW ============
                    was_on_hold = str(current_state).lower() == "on hold"
                    
                    cleaned_body = clean_user_reply(body, number) if body else ""
                    comment_to_post = cleaned_body.strip() if cleaned_body.strip() else "(Empty reply received)"
                    
                    comment_prefix = f"Customer reply via email on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}:\n\n"
                    full_comment = comment_prefix + comment_to_post
                    
                    appended = False
                    try:
                        post_incident_comment(sys_id, full_comment)
                        appended = True
                        app.logger.info(f"✓ Comment appended to incident {number}")
                        
                        # Process attachments for existing incident
                        attachment_result = process_and_upload_attachments(
                            service,
                            msg["id"],
                            msg.get("payload", {}),
                            sys_id
                        )
                        
                        if attachment_result['count'] > 0:
                            app.logger.info(f"✓ Attachments: {attachment_result['successful']}/{attachment_result['count']} uploaded")
                            
                    except Exception as e:
                        app.logger.error(f"Failed to post comment: {e}")
                    
                    state_changed = False
                    if was_on_hold and appended:
                        try:
                            if set_incident_to_in_progress(sys_id):
                                state_changed = True
                                current_state = "In Progress"
                                app.logger.info(f"✓ Incident {number} moved to In Progress")
                        except Exception as e:
                            app.logger.error(f"Error changing state: {e}")
                    
                    if was_on_hold:
                        recent_work_notes = []
                        try:
                            recent_work_notes = get_recent_work_notes(sys_id, limit=5)
                        except Exception as e:
                            app.logger.warning(f"Failed to fetch work notes: {e}")
                        
                        lines = [
                            f"Thank you for your reply regarding incident {number}.",
                            "",
                            "✓ Your information has been added to the incident comments.",
                        ]
                        
                        if state_changed:
                            lines += [
                                "✓ The incident has been moved from **On Hold** to **In Progress**.",
                                "",
                                "Our support team will continue working on your request."
                            ]
                        
                        lines += [
                            "",
                            "=== Recent Updates from Support Team (Work Notes) ==="
                        ]
                        
                        if recent_work_notes:
                            for i, note in enumerate(recent_work_notes, 1):
                                if note.get("value"):
                                    lines.append(f"\n{i}. {note['value'].strip()}")
                                    meta = []
                                    if note.get("sys_created_on"):
                                        meta.append(note['sys_created_on'])
                                    if note.get("sys_created_by"):
                                        meta.append(f"by {note['sys_created_by']}")
                                    if meta:
                                        lines.append(f"   ({' | '.join(meta)})")
                        else:
                            lines.append("\n(No recent work notes found)")
                        
                        reply_text = "\n".join(lines)
                    else:
                        lines = [
                            f"Thank you for your update on incident {number}.",
                            "",
                            "Your information has been added to the incident.",
                            "",
                            f"Current Status: **{current_state}**",
                            "",
                            "Our support team is actively working on your request and will update you shortly."
                        ]
                        reply_text = "\n".join(lines)
                    
                    try:
                        reply_subject = format_status_subject(number)
                        send_email(service, sender, reply_subject, reply_text)
                        app.logger.info(f"✓ Reply email sent for incident {number}")
                    except Exception as e:
                        app.logger.error(f"Failed to send reply email: {e}")
                    
                    outcome = {
                        "from": sender,
                        "subject": subject,
                        "body": body,
                        "decision": {"action": "EXISTING_INCIDENT_UPDATED"},
                        "result": {
                            "type": "INCIDENT_COMMENT_ADDED",
                            "incident_number": number,
                            "sys_id": sys_id,
                            "previous_state": existing_incident.get("state"),
                            "current_state": current_state,
                            "state_changed_to_progress": state_changed,
                            "comment_appended": appended,
                            "was_on_hold": was_on_hold,
                        },
                        "response_mail": reply_text
                    }
                    
                    try:
                        service.users().messages().modify(
                            userId="me",
                            id=msg["id"],
                            body={"removeLabelIds": ["UNREAD"]}
                        ).execute()
                    except Exception as mark_err:
                        outcome["mark_read_error"] = str(mark_err)
                    
                    results.append(outcome)
                    continue
                
                # ============ STEP 5: NO EXISTING INCIDENT - ML DECISION ============
                decision = generate_response(body, threshold=0.60)
                
                if decision["action"] == "AUTO_RESPONSE":
                    reply_text = decision["response"]
                    send_email(service, sender, f"Re: {subject}", reply_text)
                    
                    outcome = {
                        "from": sender,
                        "subject": subject,
                        "body": body,
                        "decision": decision,
                        "result": {"type": "AUTO_RESPONSE"},
                        "response_mail": reply_text
                    }
                
                elif decision["action"] == "REQUEST_TEMPLATE":
                    template = get_information_template()
                    template_subject = format_template_request_subject(subject)
                    
                    try:
                        send_email(service, sender, template_subject, template, is_html=True)
                        
                        if thread_id:
                            _mark_template_pending(thread_id, sender, subject)
                        
                        app.logger.info(f"✓ Template sent to {sender}")
                        
                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "decision": decision,
                            "result": {
                                "type": "TEMPLATE_REQUESTED",
                                "thread_id": thread_id,
                            },
                            "response_mail": template
                        }
                        
                    except Exception as e:
                        app.logger.error(f"Failed to send template: {e}")
                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "decision": decision,
                            "error": str(e),
                            "result": {"type": "TEMPLATE_REQUEST_FAILED"},
                        }
                
                try:
                    service.users().messages().modify(
                        userId="me",
                        id=msg["id"],
                        body={"removeLabelIds": ["UNREAD"]}
                    ).execute()
                except Exception as mark_err:
                    outcome["mark_read_error"] = str(mark_err)
                
                results.append(outcome)
            
            except Exception as per_msg_err:
                app.logger.exception("Failed processing message: %s", ref.get("id"))
                results.append({
                    "message_id": ref.get("id"),
                    "error": str(per_msg_err),
                    "result": {"type": "PROCESSING_FAILED"}
                })
        
        if not next_token:
            break
    
    return {"items": results, "blocked_count": blocked_count,"besafe_count": besafe_count}

# ---------------- ROUTES ---------------- #
@app.route("/")
def home():
    if "credentials" not in session:
        return render_template_string("""
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <title>Email Bot</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 24px; }
              a.button {
                display: inline-block;
                padding: 10px 16px;
                background: #2563eb;
                color: #fff;
                text-decoration: none;
                border-radius: 6px;
              }
              a.button:hover { background: #1d4ed8; }
              p { margin-bottom: 16px; }
            </style>
          </head>
          <body>
            <h3>Email Bot</h3>
            <p>You are not logged in.</p>
            <a class="button" href="/login">Login with Google</a>
          </body>
        </html>
        """)

    return render_template_string("""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Email Bot</title>
        <style>
          :root {
            --card-bg: #ffffff;
            --card-border: #e5e7eb;
            --box-bg: #f9fafb;
            --box-border: #e5e7eb;
            --text-muted: #6b7280;
          }
          body { font-family: Arial, sans-serif; margin: 24px; }
          header { display: flex; align-items: center; gap: 10px; }
          h3 { margin: 0; }
          #status { margin-left: auto; color: var(--text-muted); }
          .controls { margin-top: 10px; display: flex; gap: 10px; }

          button.btn, a.btn {
            padding: 8px 14px; background: #2563eb; color: #fff; border: none;
            border-radius: 6px; cursor: pointer; text-decoration: none; display: inline-block;
          }
          button.btn:hover, a.btn:hover { background: #1d4ed8; }
          button.btn:disabled { opacity: 0.6; cursor: default; }

          .results { margin-top: 16px; display: grid; gap: 16px; }
          .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .subject { font-weight: bold; margin-bottom: 10px; }
          .box {
            background: var(--box-bg);
            border: 1px solid var(--box-border);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            white-space: pre-wrap;
            line-height: 1.4;
          }
          .box:last-child { margin-bottom: 0; }
          .box-title {
            font-size: 12px; color: var(--text-muted); margin-bottom: 6px;
            text-transform: uppercase; letter-spacing: 0.04em;
          }
        </style>
      </head>
      <body>
        <header>
          <h3>Email Bot</h3>
          <div id="status">Checking unread mails...</div>
        </header>

        <div class="controls">
          <button id="btnRefresh" class="btn">Refresh Inbox</button>
          <a
            class="btn"
            href="https://dev202768.service-now.com/now/nav/ui/classic/params/target/incident_list.do%3Fsysparm_userpref_module%3D4fed4395c0a8016400fcf06c27b1%253Dtrue%255EEQ%26active%3Dtrue%26sysparm_clear_stack%3Dtrue"
            target="_blank">View Incidents
          </a>
        </div>

        <div class="results" id="results"></div>

        <script>
          const statusEl = document.getElementById('status');
          const resultsEl = document.getElementById('results');
          const btnRefresh = document.getElementById('btnRefresh');
          let pollInProgress = false;

          async function checkUnread() {
            if (pollInProgress) return;
            pollInProgress = true;
            try {
              btnRefresh.disabled = true;
              statusEl.textContent = 'Checking unread mails...';
              resultsEl.innerHTML = '';
              const resp = await fetch('/unread');
              if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
              const data = await resp.json();
              const items = (data && data.result) ? data.result : [];
              const blockedCount = data.blocked_count || 0;

              const summary = document.createElement('div');
              summary.className = 'card';
              //summary.textContent = `Blocked emails (by domain policy): ${blockedCount}`;
              resultsEl.appendChild(summary);

              if (!items.length) { 
                statusEl.textContent = 'No new mails.'; 
                return; 
              }
              statusEl.textContent = '';

              for (const it of items) {
                if (it.result && it.result.type === 'SKIPPED') continue;

                const subject = it.subject || '(No Subject)';
                const body = it.body || '';
                const response = it.response_mail || '';
                const card = document.createElement('div');
                card.className = 'card';
                const subjEl = document.createElement('div');
                subjEl.className = 'subject';
                subjEl.textContent = `Subject: ${subject}`;
                card.appendChild(subjEl);
                const bodyBox = document.createElement('div');
                bodyBox.className = 'box';
                bodyBox.innerHTML = `<div class="box-title"><b style="color:blue">Received Email Body</b></div>${escapeHtml(body)}`;
                card.appendChild(bodyBox);
                const respBox = document.createElement('div');
                respBox.className = 'box';
                respBox.innerHTML = `<div class="box-title"><b style="color:blue">Response Email</b></div>${escapeHtml(response)}`;
                card.appendChild(respBox);
                resultsEl.appendChild(card);
              }
            } catch (err) {
              console.error(err);
              statusEl.textContent = 'Error: ' + err.message;
            } finally {
              btnRefresh.disabled = false;
              pollInProgress = false;
            }
          }

          window.addEventListener('DOMContentLoaded', () => {
            checkUnread();
            setInterval(checkUnread, 10000);
          });

          btnRefresh.addEventListener('click', checkUnread);

          function escapeHtml(str) {
            if (!str) return '';
            return str
              .replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;')
              .replace(/'/g, '&#039;')
              .replace(/\\n/g, '<br/>');
          }
        </script>
      </body>
    </html>
    """)

@app.route("/login")
def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for("oauth2callback", _external=True),
    )
    auth_url, state = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true",
    )
    session["state"] = state
    return redirect(auth_url)

@app.route("/oauth2callback")
def oauth2callback():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=session.get("state"),
        redirect_uri=url_for("oauth2callback", _external=True),
    )
    flow.fetch_token(authorization_response=request.url)
    creds = flow.credentials
    session["credentials"] = creds.to_json()
    return redirect(url_for("home"))

@app.route("/unread", methods=["GET"])
def unread():
    creds_json = session.get("credentials")
    if not creds_json:
        return jsonify({"ok": False, "error": "Not authenticated. Please login."}), 401

    try:
        info = json.loads(creds_json)
        creds = Credentials.from_authorized_user_info(info, SCOPES)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            session["credentials"] = creds.to_json()

        service = build("gmail", "v1", credentials=creds)
        proc = process_unread_emails(service)
        return jsonify({
            "ok": True,
            "blocked_count": proc.get("blocked_count", 0),
            "result": sanitize_for_json(proc.get("items", []))
        })

    except Exception as e:
        app.logger.exception("Unread processing failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/ping")
def ping():
    return "pong", 200

@app.errorhandler(404)
def not_found(e):
    return f"404 Not Found: {request.path}", 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)