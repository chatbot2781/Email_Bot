
import re
import requests

class ServiceNowService:
    def __init__(self, instance: str, user: str, password: str, fields: str = None):
        self.instance = instance
        self.user = user
        self.password = password
        self.fields = fields or "number,short_description,sys_id,priority,opened_at"

    def _safe_trim(self, text: str, max_len: int = 160) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_len]

    def create_incident(
        self,
        subject: str,
        body: str,
        priority: str = "3",
        correlation_id: str = None
    ):
        if not self.instance or not self.user or not self.password:
            raise RuntimeError("ServiceNow credentials missing: set SN_INSTANCE, SN_USER, SN_PASSWORD env vars.")

        base_url = f"https://{self.instance}.service-now.com"
        url = f"{base_url}/api/now/v1/table/incident"

        params = {"sysparm_fields": self.fields, "sysparm_exclude_reference_link": "true"}
        payload = {
            "short_description": self._safe_trim(subject or "No subject"),
            "description": body or "No description",
            "priority": str(priority or "3"),
        }
        if correlation_id:
            payload["correlation_id"] = correlation_id

        resp = requests.post(url, params=params, json=payload, auth=(self.user, self.password), timeout=30)
        resp.raise_for_status()
        result = (resp.json() or {}).get("result", {}) or {}

        if "number" not in result or "sys_id" not in result:
            raise ValueError(f"Unexpected ServiceNow response: {resp.text}")

        return result
