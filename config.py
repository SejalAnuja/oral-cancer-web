import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "static/heatmaps")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

MODEL_PATH = os.path.join(BASE_DIR, "model/oral_cancer_cnn_updated.h5")

DATABASE = os.path.join(BASE_DIR, "database.db")

SECRET_KEY = "super_secure_secret_key_change_this"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
