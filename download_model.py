import gdown
import os

MODEL_PATH = "model/oral_cancer_cnn_updated.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        print("Downloading model...")
        gdown.download(
            "https://drive.google.com/uc?id=1HmSdGj2_QIYzqN09RtVPSe0iPG7cOczA",
            MODEL_PATH,
            quiet=False
        )
        print("Model downloaded!")

download_model()