from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
model = joblib.load(MODEL_PATH)

# In-memory state
queue         = []
token_counter = 0
live_crowd    = {"count": 0, "updated_at": None}

# Admin password
ADMIN_PASSWORD = "Temple123"

REQUIRED_FIELDS = ["time_hour", "day_of_week", "festival", "weather", "crowd_count"]


def validate_input(data):
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"
    if not (0 <= int(data["time_hour"]) <= 23):
        return False, "'time_hour' must be 0-23"
    if not (0 <= int(data["day_of_week"]) <= 6):
        return False, "'day_of_week' must be 0-6"
    if int(data["festival"]) not in (0, 1):
        return False, "'festival' must be 0 or 1"
    if int(data["crowd_count"]) < 0:
        return False, "'crowd_count' must be non-negative"
    return True, ""


# ── Serve frontend ────────────────────────────────────────────────
@app.route('/app')
def serve_frontend():
    return send_from_directory('.', 'index.html')


# ── Serve admin panel ─────────────────────────────────────────────
@app.route('/admin')
def serve_admin():
    return send_from_directory('.', 'admin.html')


# ── Home ──────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"message": "Temple Queue API v3.0 running"})


# ── Health ────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


# ── Predict ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"status": "failed", "error": "JSON body required"}), 400

    is_valid, err = validate_input(data)
    if not is_valid:
        return jsonify({"status": "failed", "error": err}), 400

    try:
        prediction   = model.predict([[
            int(data["time_hour"]),
            int(data["day_of_week"]),
            int(data["festival"]),
            int(data["weather"]),
            int(data["crowd_count"])
        ]])
        waiting_time = int(prediction[0])
        crowd_level  = "Low" if waiting_time <= 15 else "Moderate" if waiting_time <= 40 else "High"

        return jsonify({
            "status": "success",
            "waiting_time_minutes": waiting_time,
            "crowd_level": crowd_level
        }), 200

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


# ── YOLOv8 crowd update ───────────────────────────────────────────
@app.route("/crowd/update", methods=["POST"])
def crowd_update():
    data = request.get_json()
    if not data or "crowd_count" not in data:
        return jsonify({"status": "failed", "error": "crowd_count required"}), 400

    live_crowd["count"]      = int(data["crowd_count"])
    live_crowd["updated_at"] = datetime.now().strftime("%I:%M:%S %p")

    return jsonify({
        "status": "success",
        "crowd_count": live_crowd["count"],
        "updated_at":  live_crowd["updated_at"]
    }), 200


# ── Live crowd ────────────────────────────────────────────────────
@app.route("/crowd/live", methods=["GET"])
def crowd_live():
    return jsonify({
        "status":     "success",
        "count":      live_crowd["count"],
        "updated_at": live_crowd["updated_at"] or "Not yet updated"
    }), 200


# ── Join queue ────────────────────────────────────────────────────
@app.route("/queue/join", methods=["POST"])
def queue_join():
    global token_counter
    data = request.get_json()

    if not data or not data.get("name", "").strip():
        return jsonify({"status": "failed", "error": "Name is required"}), 400

    name     = data["name"].strip()
    existing = next((p for p in queue if p["name"].lower() == name.lower()
                     and p["status"] == "waiting"), None)

    if existing:
        ahead = sum(1 for p in queue if p["status"] == "waiting" and p["token"] < existing["token"])
        return jsonify({
            "status":        "already_joined",
            "message":       f"You are already in the queue, {name}!",
            "token":         existing["token"],
            "position":      ahead + 1,
            "total_waiting": sum(1 for p in queue if p["status"] == "waiting"),
            "joined_at":     existing["joined_at"]
        }), 200

    token_counter += 1
    entry = {
        "token":     token_counter,
        "name":      name,
        "joined_at": datetime.now().strftime("%I:%M %p"),
        "status":    "waiting"
    }
    queue.append(entry)
    ahead = sum(1 for p in queue if p["status"] == "waiting" and p["token"] < token_counter)

    return jsonify({
        "status":        "success",
        "message":       f"Welcome, {name}!",
        "token":         token_counter,
        "position":      ahead + 1,
        "total_waiting": sum(1 for p in queue if p["status"] == "waiting"),
        "joined_at":     entry["joined_at"]
    }), 200


# ── Queue status ──────────────────────────────────────────────────
@app.route("/queue/status", methods=["GET"])
def queue_status():
    waiting = [p for p in queue if p["status"] == "waiting"]
    return jsonify({
        "status":        "success",
        "total_waiting": len(waiting),
        "queue": [
            {"position": i+1, "token": p["token"],
             "name": p["name"], "joined_at": p["joined_at"]}
            for i, p in enumerate(waiting)
        ]
    }), 200


# ── Token status ──────────────────────────────────────────────────
@app.route("/queue/token/<int:token>", methods=["GET"])
def token_status(token):
    entry = next((p for p in queue if p["token"] == token), None)
    if not entry:
        return jsonify({"status": "failed", "error": "Token not found"}), 404

    waiting  = [p for p in queue if p["status"] == "waiting"]
    position = next((i+1 for i, p in enumerate(waiting) if p["token"] == token), None)

    if position is None:
        return jsonify({"status": "success", "token": token,
                        "name": entry["name"], "queue_status": "done"}), 200

    return jsonify({
        "status":                 "success",
        "token":                  token,
        "name":                   entry["name"],
        "position":               position,
        "total_waiting":          len(waiting),
        "people_ahead":           position - 1,
        "estimated_wait_minutes": (position - 1) * 5,
        "joined_at":              entry["joined_at"],
        "queue_status":           "waiting"
    }), 200


# ── Admin: verify password ────────────────────────────────────────
@app.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json()
    if not data or data.get("password") != ADMIN_PASSWORD:
        return jsonify({"status": "failed", "error": "Wrong password"}), 401
    return jsonify({"status": "success", "message": "Login successful"}), 200


# ── Admin: call next token ────────────────────────────────────────
@app.route("/admin/call-next", methods=["POST"])
def call_next():
    data = request.get_json()
    if not data or data.get("password") != ADMIN_PASSWORD:
        return jsonify({"status": "failed", "error": "Unauthorized"}), 401

    waiting = [p for p in queue if p["status"] == "waiting"]
    if not waiting:
        return jsonify({"status": "failed", "error": "No one in queue"}), 400

    next_person = waiting[0]
    next_person["status"] = "done"
    next_person["called_at"] = datetime.now().strftime("%I:%M %p")

    remaining = sum(1 for p in queue if p["status"] == "waiting")

    return jsonify({
        "status":       "success",
        "called_token": next_person["token"],
        "called_name":  next_person["name"],
        "remaining":    remaining
    }), 200


# ── Admin: mark specific token as done ───────────────────────────
@app.route("/admin/mark-done", methods=["POST"])
def mark_done():
    data = request.get_json()
    if not data or data.get("password") != ADMIN_PASSWORD:
        return jsonify({"status": "failed", "error": "Unauthorized"}), 401

    token = data.get("token")
    entry = next((p for p in queue if p["token"] == token), None)
    if not entry:
        return jsonify({"status": "failed", "error": "Token not found"}), 404

    entry["status"] = "done"
    entry["called_at"] = datetime.now().strftime("%I:%M %p")

    return jsonify({"status": "success", "message": f"Token {token} marked as done"}), 200


# ── Admin: reset queue ────────────────────────────────────────────
@app.route("/admin/reset", methods=["POST"])
def reset_queue():
    global queue, token_counter
    data = request.get_json()
    if not data or data.get("password") != ADMIN_PASSWORD:
        return jsonify({"status": "failed", "error": "Unauthorized"}), 401

    queue         = []
    token_counter = 0

    return jsonify({"status": "success", "message": "Queue reset successfully"}), 200


# ── Admin: update crowd manually ─────────────────────────────────
@app.route("/admin/crowd", methods=["POST"])
def admin_crowd():
    data = request.get_json()
    if not data or data.get("password") != ADMIN_PASSWORD:
        return jsonify({"status": "failed", "error": "Unauthorized"}), 401

    live_crowd["count"]      = int(data.get("count", 0))
    live_crowd["updated_at"] = datetime.now().strftime("%I:%M:%S %p")

    return jsonify({
        "status":     "success",
        "count":      live_crowd["count"],
        "updated_at": live_crowd["updated_at"]
    }), 200


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
