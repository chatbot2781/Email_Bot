
from flask import Blueprint, render_template, session

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def home():
    if "credentials" not in session:
        return render_template("home_logged_out.html")
    return render_template("home_logged_in.html")
