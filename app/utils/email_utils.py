
import re
import base64
import numpy as np

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def decode_base64url(data: str | None) -> str:
    if not data:
        return ""
    data += "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    except Exception:
        return ""

def strip_html(html: str) -> str:
    """Remove basic HTML tags and collapse whitespace."""
    if not html:
        return ""
    html = re.sub(r"<.*?>", "", html, flags=re.S)
    return re.sub(r"\s+", " ", html).strip()

def extract_body(payload: dict) -> str:
    """Extract text body from Gmail message payload."""
    body = ""
    if "parts" in payload:
        for part in payload.get("parts", []):
            if part.get("mimeType") in ["text/plain", "text/html"]:
                data = part.get("body", {}).get("data")
                if data:
                    body = decode_base64url(data)
    else:
        body = decode_base64url(payload.get("body", {}).get("data"))
    return strip_html(body)

def get_header(headers: list, name: str, default: str = "") -> str:
    """Safely fetch a header value by name (case-insensitive)."""
    target = name.lower()
    for h in headers or []:
        if h.get("name", "").lower() == target:
            return h.get("value", default)
    return default

