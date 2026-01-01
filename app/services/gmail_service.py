
from email.mime.text import MIMEText
import base64
from ..utils.email_utils import extract_body, get_header

def send_email(service, to_email: str, subject: str, body: str):
    message = MIMEText(body)
    message["to"] = to_email
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()

def process_unread_emails(service, ml_service, sn_service, logger=None):
    """
    Process all unread emails (pagination supported).
    Returns a list of outcomes per message.
    """
    results = []
    next_token = None

    while True:
        list_kwargs = {"userId": "me", "labelIds": ["UNREAD"], "maxResults": 50}
        if next_token:
            list_kwargs["pageToken"] = next_token

        res = service.users().messages().list(**list_kwargs).execute()
        messages = res.get("messages", [])
        next_token = res.get("nextPageToken")

        if not messages:
            break

        for ref in messages:
            try:
                msg = service.users().messages().get(userId="me", id=ref["id"], format="full").execute()

                headers = msg.get("payload", {}).get("headers", [])
                sender = get_header(headers, "From", default="unknown@example.com")
                subject = get_header(headers, "Subject", default="(No Subject)")
                body = extract_body(msg.get("payload", {}))

                decision = ml_service.generate_response(body)

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

                else:
                    try:
                        incident = sn_service.create_incident(
                            subject=subject,
                            body=body,
                            priority="3",
                            correlation_id=msg.get("id"),
                        )

                        reply_text = (
                            f"Your request has been received and an incident "
                            f"{incident.get('number')} has been created.\n\n"
                            f"Short description: {incident.get('short_description')}\n"
                            f"Priority: {incident.get('priority')}\n"
                            f"Opened at: {incident.get('opened_at')}\n\n"
                            "Our support team will contact you shortly."
                        )

                        send_email(service, sender, f"Re: {subject}", reply_text)

                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "decision": decision,
                            "result": {
                                "type": "INCIDENT_CREATED",
                                "incident_number": incident.get("number"),
                                "sys_id": incident.get("sys_id"),
                                "priority": incident.get("priority"),
                            },
                            "response_mail": reply_text
                        }

                    except Exception as e:
                        fallback_text = (
                            "We attempted to create an incident for your request but ran into an issue. "
                            "Please try again later or contact support."
                        )
                        send_email(service, sender, f"Re: {subject}", fallback_text)

                        outcome = {
                            "from": sender,
                            "subject": subject,
                            "body": body,
                            "decision": decision,
                            "error": str(e),
                            "result": {"type": "INCIDENT_FAILED"},
                            "response_mail": fallback_text
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

            except Exception as per_msg_err:
                if logger:
                    logger.exception("Failed processing message: %s", ref.get("id"))
                results.append({
                    "message_id": ref.get("id"),
                    "error": str(per_msg_err),
                    "result": {"type": "PROCESSING_FAILED"}
                })

        if not next_token:
            break

    return results
