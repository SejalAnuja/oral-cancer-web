from download_model import download_model
download_model()
from flask import Flask, render_template, request, send_file, redirect, session
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tensorflow as tf
import numpy as np
import cv2
import os
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from gradcam import make_gradcam_heatmap, overlay_heatmap


# =========================
# Flask setup
# =========================
app = Flask(__name__)

app.secret_key = "super_secure_secret_key"

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# =========================
# Load model
# =========================
model = tf.keras.models.load_model(
    "model/oral_cancer_cnn_updated.h5",
    compile=False
)

# Warm up the model
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model(dummy, training=False)

print("Model loaded successfully")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# =========================
# Database init
# =========================
def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result TEXT,
            confidence REAL,
            cancer_prob REAL,
            normal_prob REAL,
            image_path TEXT,
            heatmap_path TEXT,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()


# =========================
# Login Route
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE email=?", (email,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect("/dashboard")

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# =========================
# Signup Route
# =========================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        if not name or not email or not password:
            return render_template("signup.html", error="All fields are required")

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, hashed_password)
            )
            conn.commit()
            conn.close()
            return redirect("/")
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Email already registered")

    return render_template("signup.html")


# =========================
# Dashboard Route
# =========================
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/")
    return render_template("dashboard.html")


# =========================
# Prediction Route
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return redirect("/")

    if "image" not in request.files:
        return render_template("dashboard.html", result="No file uploaded")

    file = request.files["image"]

    if file.filename == "":
        return render_template("dashboard.html", result="No file selected")

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename).replace("\\", "/")
    heatmap_path = os.path.join(HEATMAP_FOLDER, filename).replace("\\", "/")

    file.save(filepath)

    # -------------------------------------------------------
    # Preprocess — BGR->RGB to match training data convention
    # -------------------------------------------------------
    img_bgr = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # FIX: convert to RGB
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)

    # -------------------------------------------------------
    # Prediction (sigmoid output: 0=Normal, 1=Cancer)
    # -------------------------------------------------------
    prediction = float(model(img_input, training=False)[0][0])

    cancer_prob = round(prediction * 100, 2)
    normal_prob = round((1 - prediction) * 100, 2)

    if prediction > 0.5:
        result = "Cancer"
        confidence = cancer_prob
    else:
        result = "Normal"
        confidence = normal_prob

    confidence = round(confidence, 2)

    # -------------------------------------------------------
    # GradCAM heatmap
    # -------------------------------------------------------
    heatmap = make_gradcam_heatmap(img_input, model)
    overlay_heatmap(filepath, heatmap, heatmap_path)

    # -------------------------------------------------------
    # Save to database
    # -------------------------------------------------------
    conn = get_db()
    cursor = conn.execute("""
        INSERT INTO predictions
        (result, confidence, cancer_prob, normal_prob, image_path, heatmap_path, date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        result, confidence, cancer_prob, normal_prob,
        filepath, heatmap_path,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return render_template(
        "dashboard.html",
        result=result,
        confidence=confidence,
        cancer_prob=cancer_prob,
        normal_prob=normal_prob,
        image_path=filepath,
        heatmap_path=heatmap_path,
        prediction_id=prediction_id
    )


# =========================
# History Route
# =========================
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/")

    conn = get_db()

    page = request.args.get("page", 1, type=int)
    per_page = 5
    offset = (page - 1) * per_page

    total_scans = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    data = conn.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """, (per_page, offset)).fetchall()

    cancer_count = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE result='Cancer'"
    ).fetchone()[0]

    detection_rate = round((cancer_count / total_scans) * 100, 2) if total_scans > 0 else 0

    avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
    avg_conf = round(avg_conf, 2) if avg_conf else 0

    conn.close()

    total_pages = (total_scans + per_page - 1) // per_page

    return render_template(
        "history.html",
        data=data,
        page=page,
        total_pages=total_pages,
        total_scans=total_scans,
        detection_rate=detection_rate,
        avg_confidence=avg_conf
    )


# =========================
# PDF Report Route
# =========================
@app.route("/download-report/<int:id>")
def download_report(id):
    conn = sqlite3.connect("database.db")
    cursor = conn.execute("""
        SELECT result, confidence, cancer_prob, normal_prob,
               image_path, heatmap_path, date
        FROM predictions WHERE id=?
    """, (id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return "Report not found"

    result, confidence, cancer_prob, normal_prob, image_path, heatmap_path, date = row

    pdf_path = os.path.join(REPORT_FOLDER, f"report_{id}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(150, height - 50, "Oral Cancer Detection Report")

    # Details
    c.setFont("Helvetica", 12)
    y = height - 100

    c.drawString(50, y, f"Date: {date}");             y -= 25
    c.drawString(50, y, f"Classification: {result}"); y -= 25
    c.drawString(50, y, f"Cancer Probability: {cancer_prob}%"); y -= 25
    c.drawString(50, y, f"Normal Probability: {normal_prob}%"); y -= 40

    # Recommendation
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Clinical Recommendation:")
    y -= 25

    c.setFont("Helvetica", 12)
    if result == "Cancer" and cancer_prob >= 85:
        recommendation = "High-risk lesion detected. Immediate specialist consultation required."
    elif result == "Cancer" and cancer_prob >= 50:
        recommendation = "Suspicious lesion detected. Specialist evaluation recommended."
    else:
        recommendation = "No cancer detected. Routine monitoring advised."

    c.drawString(50, y, recommendation)

    # Images
    y -= 200
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y + 160, "Original Image")
    try:
        c.drawImage(image_path, 50, y, width=200, height=150)
    except Exception:
        pass

    c.drawString(300, y + 160, "GradCAM Heatmap")
    try:
        c.drawImage(heatmap_path, 300, y, width=200, height=150)
    except Exception:
        pass

    # Footer
    c.setFont("Helvetica", 10)
    c.drawString(50, 50, "Generated by OralScan AI System")
    c.drawString(50, 35, "For clinical assistance only")

    c.save()

    return send_file(pdf_path, as_attachment=True)


# =========================
# Logout Route
# =========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)