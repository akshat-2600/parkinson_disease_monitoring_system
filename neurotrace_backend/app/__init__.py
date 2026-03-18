"""
app/__init__.py
Flask application factory.

What this does end-to-end:
  1.  Creates Flask app pointing to /templates and /static folders
  2.  Registers every extension (DB, JWT, CORS, Bcrypt)
  3.  Mounts all prediction API blueprints under /api/*
  4.  Serves index.html via render_template for every browser URL
      so `python run.py` → open http://localhost:5000 → full app loads.

URL layout:
  GET  /                          → render_template("index.html")
  GET  /dashboard, /login …       → render_template("index.html")  (SPA routes)
  POST /api/auth/login            → JSON API
  POST /api/voice/predict         → JSON API
  POST /api/fusion/realtime_predict → JSON API
  GET  /api/fusion/dashboard/<id> → JSON API
  … (all /api/* routes are pure JSON, no templates)
"""
import os
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from flask_bcrypt import Bcrypt

from config import get_config

# ── Extension singletons (imported by every sub-module) ───────
db      = SQLAlchemy()
migrate = Migrate()
jwt     = JWTManager()
bcrypt  = Bcrypt()


def create_app(config_class=None):
    """Application factory — call once in run.py."""

    # Flask needs to know where templates/ and static/ live.
    # Both sit one level above app/, i.e. at the project root.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    app = Flask(
        __name__,
        template_folder=os.path.join(root, "templates"),
        static_folder=os.path.join(root, "static"),
        static_url_path="/static",
    )

    cfg = config_class or get_config()
    app.config.from_object(cfg)

    # ── Extensions ───────────────────────────────────────────
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    bcrypt.init_app(app)
    CORS(
        app,
        resources={r"/api/*": {"origins": "*"}},
        supports_credentials=True,
    )
    _register_jwt_handlers(jwt)

    # ── API Blueprints (all prefixed /api/) ───────────────────
    with app.app_context():
        from app.api.auth           import auth_bp
        from app.api.voice          import voice_bp
        from app.api.clinical       import clinical_bp
        from app.api.mri            import mri_bp
        from app.api.spiral         import spiral_bp
        from app.api.motor          import motor_bp
        from app.api.timeseries     import timeseries_bp
        from app.api.fusion         import fusion_bp
        from app.api.explainability import explain_bp
        from app.api.patients       import patients_bp

        # ----- Authentication -----
        app.register_blueprint(auth_bp,       url_prefix="/api/auth")

        # ----- Per-modality prediction -----
        app.register_blueprint(voice_bp,      url_prefix="/api/voice")
        app.register_blueprint(clinical_bp,   url_prefix="/api/clinical")
        app.register_blueprint(mri_bp,        url_prefix="/api/mri")
        app.register_blueprint(spiral_bp,     url_prefix="/api/spiral")
        app.register_blueprint(motor_bp,      url_prefix="/api/motor")
        app.register_blueprint(timeseries_bp, url_prefix="/api/timeseries")

        # ----- Fusion engine (dashboard, history, recommendations …) -----
        app.register_blueprint(fusion_bp,     url_prefix="/api/fusion")

        # ----- Explainability (SHAP / LIME / Grad-CAM) -----
        app.register_blueprint(explain_bp,    url_prefix="/api/explain")

        # ----- Patient & Report management -----
        app.register_blueprint(patients_bp,   url_prefix="/api/patients")

        # Create DB tables on first run
        db.create_all()

    # ── System routes ─────────────────────────────────────────

    @app.get("/health")
    def health():
        """Docker / load-balancer health check — returns JSON, no auth needed."""
        return jsonify({"status": "ok", "service": "NeuroTrace", "version": "1.0.0"})

    # ── Frontend (SPA) routes — serve index.html ──────────────
    # Flask renders index.html and injects the API base URL so JS
    # always points at the correct server without any hardcoding.

    SPA_ROUTES = [
        "/",
        "/login",
        "/signup",
        "/dashboard",
        "/realtime",
        "/history",
        "/explanation",
        "/recommendations",
    ]

    def _render_spa():
        return render_template(
            "index.html",
            api_base  = _resolve_api_base(),
            flask_env = app.config.get("FLASK_ENV", "development"),
            version   = "1.0.0",
        )

    # Register each SPA path explicitly
    for _path in SPA_ROUTES:
        # Use a unique endpoint name per route
        endpoint = "spa_" + _path.strip("/").replace("/", "_") or "spa_root"
        app.add_url_rule(_path, endpoint=endpoint, view_func=_render_spa)

    # Catch-all 404: return JSON for /api/* errors, SPA for everything else
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "API endpoint not found", "path": request.path}), 404
        return _render_spa()

    return app


# ── Helpers ───────────────────────────────────────────────────

def _resolve_api_base():
    """
    Determine the API base URL injected into the HTML template.

    - Development:  empty string → JS uses same-origin relative paths (/api/…)
    - Production:   set API_BASE_URL env var to override (e.g. https://api.yourdomain.com)
    """
    return os.getenv("API_BASE_URL", "")


def _register_jwt_handlers(jwt_manager):
    """Standardise all JWT error responses as JSON."""

    @jwt_manager.expired_token_loader
    def expired(jwt_header, jwt_data):
        return jsonify({"error": "Token has expired — please log in again"}), 401

    @jwt_manager.invalid_token_loader
    def invalid(error):
        return jsonify({"error": "Invalid authentication token"}), 401

    @jwt_manager.unauthorized_loader
    def missing(error):
        return jsonify({"error": "Authorization token required"}), 401

    @jwt_manager.revoked_token_loader
    def revoked(jwt_header, jwt_data):
        return jsonify({"error": "Token has been revoked — please log in again"}), 401