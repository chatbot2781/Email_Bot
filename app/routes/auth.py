
from flask import Blueprint, session, redirect, url_for, current_app, request
from google_auth_oauthlib.flow import Flow

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/login")
def login():
    flow = Flow.from_client_secrets_file(
        current_app.config["CLIENT_SECRETS_FILE"],
        scopes=current_app.config["SCOPES"],
        redirect_uri=url_for("auth.oauth2callback", _external=True),
    )
    auth_url, state = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true",
    )
    session["state"] = state
    return redirect(auth_url)

@auth_bp.route("/oauth2callback")
def oauth2callback():
    flow = Flow.from_client_secrets_file(
        current_app.config["CLIENT_SECRETS_FILE"],
        scopes=current_app.config["SCOPES"],
        state=session.get("state"),
        redirect_uri=url_for("auth.oauth2callback", _external=True),
    )
    flow.fetch_token(authorization_response=request.url)
    creds = flow.credentials
    session["credentials"] = creds.to_json()
    return redirect(url_for("main.home"))
