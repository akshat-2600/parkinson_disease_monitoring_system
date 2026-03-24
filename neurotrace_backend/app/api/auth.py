"""
app/api/auth.py
Authentication endpoints — signup, login, refresh, logout, me.
"""
from flask import Blueprint, request
from flask_jwt_extended import (
    create_access_token, create_refresh_token,
    jwt_required, verify_jwt_in_request, get_jwt_identity, get_jwt,
)
from app import db, bcrypt
from app.models import User, Patient, Doctor
from app.utils.response import success, error

auth_bp = Blueprint("auth", __name__)

# In-memory token blocklist (use Redis in production)
_blocklist: set = set()

# ── ADD THIS ──────────────────────────────────────────────────
from app import jwt   # import the jwt instance from your app factory

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    return jwt_payload["jti"] in _blocklist
# ─────────────────────────────────────────────────────────────


# ── Signup ────────────────────────────────────────────────────
@auth_bp.post("/signup")
def signup():
    """
    POST /auth/signup
    Body: { email, password, role, first_name, last_name,
            [patient_uid, age, gender, diagnosis, onset_year] (if patient)
            [specialisation, license_number] (if doctor) }
    """
    data = request.get_json(silent=True) or {}

    # Validation
    for field in ("email", "password", "role"):
        if not data.get(field):
            return error(f"'{field}' is required", 400)

    role = data["role"].lower()
    if role not in ("doctor", "patient"):
        return error("role must be 'doctor' or 'patient'", 400)

    if User.query.filter_by(email=data["email"]).first():
        return error("Email already registered", 409)

    # Create user
    pw_hash = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    user = User(
        email         = data["email"].lower().strip(),
        password_hash = pw_hash,
        role          = role,
        first_name    = data.get("first_name", ""),
        last_name     = data.get("last_name", ""),
    )
    db.session.add(user)
    db.session.flush()   # get user.id before committing

    # Role-specific profile
    if role == "patient":
        uid = data.get("patient_uid") or f"PT-{user.id:04d}"
        profile = Patient(
            user_id     = user.id,
            patient_uid = uid,
            age         = data.get("age"),
            gender      = data.get("gender"),
            diagnosis   = data.get("diagnosis", "Parkinson's Disease"),
            onset_year  = data.get("onset_year"),
        )
        db.session.add(profile)
    else:
        profile = Doctor(
            user_id        = user.id,
            specialisation = data.get("specialisation", "Neurology"),
            license_number = data.get("license_number", ""),
        )
        db.session.add(profile)

    db.session.commit()

    identity = {"user_id": user.id, "role": user.role}
    return success(
        data={
            "user":          user.to_dict(),
            "access_token":  create_access_token(identity=identity),
            "refresh_token": create_refresh_token(identity=identity),
        },
        message="Account created successfully",
        status=201,
    )


# ── Login ─────────────────────────────────────────────────────
@auth_bp.post("/login")
def login():
    """
    POST /auth/login
    Body: { email, password }
    """
    data = request.get_json(silent=True) or {}
    email    = data.get("email", "").lower().strip()
    password = data.get("password", "")

    if not email or not password:
        return error("email and password are required", 400)

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return error("Invalid email or password", 401)

    if not user.is_active:
        return error("Account is deactivated", 403)

    identity = str(user.id)
    additional_claims = {"role": user.role}

    # Attach profile-specific context
    if user.role == "patient" and user.patient_profile:
        additional_claims["patient_uid"] = user.patient_profile.patient_uid
    elif user.role == "doctor" and user.doctor_profile:
        additional_claims["doctor_id"] = user.doctor_profile.id

    return success(data={
        "user":          user.to_dict(),
        "access_token":  create_access_token(identity=identity, additional_claims=additional_claims),
        "refresh_token": create_refresh_token(identity=identity, additional_claims=additional_claims),
    })


# ── Refresh ───────────────────────────────────────────────────
@auth_bp.post("/refresh")
def refresh():
    """POST /auth/refresh — obtain a new access token.

    Accepts a Bearer refresh token in Authorization header.
    Falls back to an access token if refresh token is missing from client.
    """
    from flask import current_app
    from flask_jwt_extended import verify_jwt_in_request

    current_app.logger.info("/api/auth/refresh called")

    # Try with refresh token first
    try:
        verify_jwt_in_request(refresh=True)
        current_app.logger.info("Refresh token verified as refresh token")
    except Exception as refresh_exc:
        current_app.logger.warning("Refresh verify failed: %s", refresh_exc)
        # Fallback: maybe the client sent the access token by mistake
        try:
            verify_jwt_in_request(refresh=False)
            current_app.logger.info("Refresh endpoint accepted regular access token (fallback)")
        except Exception as exc:
            current_app.logger.error("Refresh endpoint failed on both refresh and access token: %s", exc)
            return error("Invalid or expired refresh token", 401)

    identity = get_jwt_identity()
    if not identity:
        current_app.logger.error("Refresh endpoint could not extract identity from token")
        return error("Invalid token identity", 401)

    user_id = int(identity)
    current_app.logger.info("Refresh succeeded for user id %s", user_id)

    # Get additional claims
    claims = get_jwt()
    additional_claims = {"role": claims.get("role")}
    if "patient_uid" in claims:
        additional_claims["patient_uid"] = claims["patient_uid"]
    if "doctor_id" in claims:
        additional_claims["doctor_id"] = claims["doctor_id"]

    access_token  = create_access_token(identity=identity, additional_claims=additional_claims)
    refresh_token = create_refresh_token(identity=identity, additional_claims=additional_claims)

    return success(data={"access_token": access_token, "refresh_token": refresh_token})


# ── Logout ────────────────────────────────────────────────────
@auth_bp.delete("/logout")
@jwt_required()
def logout():
    """DELETE /auth/logout — blacklist current token."""
    jti = get_jwt()["jti"]
    _blocklist.add(jti)
    return success(message="Logged out successfully")


# ── Me ────────────────────────────────────────────────────────
@auth_bp.get("/me")
@jwt_required()
def me():
    """GET /auth/me — return current user profile."""
    identity = get_jwt_identity()
    user_id = int(identity)
    user     = User.query.get(user_id)
    if not user:
        return error("User not found", 404)

    profile = {}
    if user.role == "patient" and user.patient_profile:
        profile = user.patient_profile.to_dict()
    elif user.role == "doctor" and user.doctor_profile:
        profile = {
            "id":             user.doctor_profile.id,
            "specialisation": user.doctor_profile.specialisation,
            "license_number": user.doctor_profile.license_number,
        }

    return success(data={"user": user.to_dict(), "profile": profile})


# ── Change password ───────────────────────────────────────────
@auth_bp.put("/change-password")
@jwt_required()
def change_password():
    identity = get_jwt_identity()
    user     = User.query.get(identity["user_id"])
    data     = request.get_json(silent=True) or {}

    if not bcrypt.check_password_hash(user.password_hash, data.get("current_password", "")):
        return error("Current password is incorrect", 401)

    new_pw = data.get("new_password", "")
    if len(new_pw) < 8:
        return error("New password must be at least 8 characters", 400)

    user.password_hash = bcrypt.generate_password_hash(new_pw).decode("utf-8")
    db.session.commit()
    return success(message="Password updated successfully")