
import os
from flask import Flask
from .config import Config
from .routes.main import main_bp
from .routes.auth import auth_bp
from .routes.api import api_bp
from .services.ml_service import MLService
from .services.servicenow_service import ServiceNowService

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    # Set to allow http for local testing (remove/disable in production)
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = app.config.get("OAUTHLIB_INSECURE_TRANSPORT", "1")

    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    # Initialize services and store them in app.extensions for reuse
    app.logger.info("Initializing ML service...")
    ml_service = MLService(model_dir=app.config["MODEL_DIR"])

    app.logger.info("Initializing ServiceNow service...")
    sn_service = ServiceNowService(
        instance=app.config.get("SN_INSTANCE"),
        user=app.config.get("SN_USER"),
        password=app.config.get("SN_PASSWORD")
    )

    app.extensions = getattr(app, "extensions", {})
    app.extensions["ml_service"] = ml_service
    app.extensions["sn_service"] = sn_service

    app.logger.info("App initialized.")
    return app
