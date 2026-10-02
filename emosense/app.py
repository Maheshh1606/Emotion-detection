"""EmoSense - real-time facial emotion detection web app (Flask + TensorFlow + OpenCV).

Run train.py first to create emotion_model.keras and labels.txt, then: python app.py
"""
import base64
import os
import sqlite3
from functools import wraps

import cv2
import numpy as np
from flask import (Flask, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-before-deploying")

DB_PATH = "emosense.db"
MODEL_PATH = "emotion_model.keras"
LABELS_PATH = "labels.txt"

EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😊",
         "neutral": "😐", "sad": "😢", "surprise": "😲"}
COLORS = {"angry": "#ef4444", "disgust": "#84cc16", "fear": "#f59e0b",
          "happy": "#22c55e", "neutral": "#94a3b8", "sad": "#3b82f6",
          "surprise": "#8b5cf6"}

# ---------- model ----------
LABELS = list(EMOJI)
model = None
if os.path.exists(LABELS_PATH):
    LABELS = open(LABELS_PATH).read().split()
if os.path.exists(MODEL_PATH):
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print(f"[EmoSense] {MODEL_PATH} not found. Run train.py first.")

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------- database ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                auto_save INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)


@app.before_request
def load_user():
    g.user = None
    if "user_id" in session:
        g.user = get_db().execute(
            "SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
        if g.user is None:
            session.clear()


@app.context_processor
def inject_globals():
    return dict(EMOJI=EMOJI, COLORS=COLORS)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# ---------- auth ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        if not name or not email or len(password) < 6:
            flash("Enter your name, an email, and a password of at least 6 characters.")
        else:
            try:
                db = get_db()
                cur = db.execute(
                    "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                    (name, email, generate_password_hash(password)))
                db.commit()
                session["user_id"] = cur.lastrowid
                return redirect(url_for("home"))
            except sqlite3.IntegrityError:
                flash("That email is already registered. Log in instead.")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        user = get_db().execute(
            "SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], request.form["password"]):
            session["user_id"] = user["id"]
            return redirect(url_for("home"))
        flash("Email or password is incorrect.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- pages ----------
@app.route("/")
@login_required
def home():
    db = get_db()
    uid = g.user["id"]
    total = db.execute("SELECT COUNT(*) FROM detections WHERE user_id = ?", (uid,)).fetchone()[0]
    top = db.execute(
        "SELECT emotion, COUNT(*) c FROM detections WHERE user_id = ? "
        "GROUP BY emotion ORDER BY c DESC LIMIT 1", (uid,)).fetchone()
    last = db.execute(
        "SELECT * FROM detections WHERE user_id = ? ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
    return render_template("home.html", total=total, top=top, last=last,
                           model_ready=model is not None)


@app.route("/detect")
@login_required
def detect():
    return render_template("detect.html", labels=LABELS, model_ready=model is not None)


@app.route("/history")
@login_required
def history():
    rows = get_db().execute(
        "SELECT * FROM detections WHERE user_id = ? ORDER BY id DESC LIMIT 200",
        (g.user["id"],)).fetchall()
    return render_template("history.html", rows=rows)


@app.post("/history/clear")
@login_required
def clear_history():
    db = get_db()
    db.execute("DELETE FROM detections WHERE user_id = ?", (g.user["id"],))
    db.commit()
    flash("History cleared.")
    return redirect(url_for("history"))


@app.route("/analytics")
@login_required
def analytics():
    db = get_db()
    rows = db.execute(
        "SELECT emotion, COUNT(*) c, AVG(confidence) a FROM detections "
        "WHERE user_id = ? GROUP BY emotion ORDER BY c DESC", (g.user["id"],)).fetchall()
    total = sum(r["c"] for r in rows)
    avg_conf = (sum(r["c"] * r["a"] for r in rows) / total) if total else 0
    return render_template("analytics.html", rows=rows, total=total, avg_conf=avg_conf)


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        name = request.form["name"].strip() or g.user["name"]
        auto_save = 1 if request.form.get("auto_save") else 0
        db = get_db()
        db.execute("UPDATE users SET name = ?, auto_save = ? WHERE id = ?",
                   (name, auto_save, g.user["id"]))
        db.commit()
        flash("Settings saved.")
        return redirect(url_for("settings"))
    return render_template("settings.html")


# ---------- prediction API ----------
@app.post("/predict")
@login_required
def predict():
    if model is None:
        return jsonify(error="Model not found. Run train.py to create emotion_model.keras."), 503
    data = request.get_json(silent=True) or {}
    try:
        raw = base64.b64decode(data["image"].split(",")[-1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return jsonify(error="The camera frame could not be read."), 400

    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return jsonify(face=False)

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = cv2.resize(gray[y:y + h, x:x + w], (48, 48)).reshape(1, 48, 48, 1).astype("float32")
    probs = model.predict(face, verbose=0)[0]
    best = int(np.argmax(probs))
    emotion, confidence = LABELS[best], round(float(probs[best]) * 100, 1)

    if data.get("save") and g.user["auto_save"]:
        db = get_db()
        db.execute("INSERT INTO detections (user_id, emotion, confidence) VALUES (?, ?, ?)",
                   (g.user["id"], emotion, confidence))
        db.commit()

    return jsonify(
        face=True, emotion=emotion, confidence=confidence,
        probs={l: round(float(p) * 100, 1) for l, p in zip(LABELS, probs)},
        box=[int(x), int(y), int(w), int(h)],
        frame_size=[int(frame.shape[1]), int(frame.shape[0])])


if __name__ == "__main__":
    init_db()
    app.run(debug=False, port=5000)
