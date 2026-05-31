import numpy as np
import cv2

IMG_SIZE = 224

# NOTE: The model is loaded once in app.py.
# utils.py provides stateless helper functions only —
# no second model load here to avoid wasting ~1 GB of RAM.


def preprocess_image(image_path):
    """
    Load, convert BGR->RGB (matching training convention),
    resize, normalise, and add batch dimension.
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # match training colour space
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    return img_input


def parse_prediction(raw_prob: float):
    """
    Convert the raw sigmoid probability to a label + confidence pair.
    raw_prob: float in [0, 1]  (1 = Cancer, 0 = Normal)
    Returns: (label: str, confidence: float, cancer_prob: float, normal_prob: float)
    """
    cancer_prob = round(raw_prob * 100, 2)
    normal_prob = round((1 - raw_prob) * 100, 2)

    if raw_prob > 0.5:
        label = "Cancer"
        confidence = cancer_prob
    else:
        label = "Normal"
        confidence = normal_prob

    return label, round(confidence, 2), cancer_prob, normal_prob