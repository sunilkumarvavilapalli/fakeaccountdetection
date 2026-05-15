from flask import Flask, render_template, request, redirect, url_for, flash, session
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import easyocr
import pandas as pd
import re
from difflib import SequenceMatcher
import numpy as np
import joblib
import os
import logging
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'


# =========================
# LOAD MODELS
# =========================
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    reader = easyocr.Reader(['en'])

    scaler_x = joblib.load("scaler.pkl")
    clf = joblib.load("fake_account_rf_lr_nb.pkl")

    # Backward-compatibility patch for models pickled across sklearn versions.
    for est in getattr(clf, "estimators_", []):
        if isinstance(est, LogisticRegression) and not hasattr(est, "multi_class"):
            est.multi_class = "auto"

    logger.info("Models loaded successfully")

except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise


# =========================
# UTILITY FUNCTIONS
# =========================

def check_profile_face(image):

    image = image.convert("RGB")

    width, height = image.size

    top_crop = image.crop((0, 0, width, int(height * 0.2)))

    text = ["Has face", "Doesn't have face"]

    inputs = clip_processor(text=text, images=top_crop, return_tensors="pt", padding=True)

    outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)

    prediction = probs.argmax().item()

    return 1 if prediction == 0 else 0


def extract_text_from_image(image):

    return reader.readtext(np.array(image), detail=0)


def extract_ordered_numbers(lines):

    numbers = []

    for line in lines:

        line = line.replace(",", "")

        if ":" in line:
            continue

        found = re.findall(r'\b\d+\b', line)

        numbers.extend([int(n) for n in found])

        if len(numbers) >= 3:
            break

    return numbers[:3]


def contains_url(text):

    url_pattern = re.compile(r"(https?://\S+|www\.\S+)")

    return 1 if any(re.search(url_pattern, line) for line in text) else 0


def parse_profile_data(lines):

    profile = {

        "UserName": None,
        "FullName": None,
        "description": "",
        "Posts": None,
        "Followers": None,
        "Following": None,

        "UserName_LetterCount": 0,
        "FullName_LetterCount": 0,

        "UserName_DigitCount": 0,
        "FullName_DigitCount": 0,

        "FullName_WordCount": 0,

        "Match": 0,

        "nums/length username": 0.0,
        "nums/length fullname": 0.0,

        "Has_URL": contains_url(lines)

    }

    numbers = extract_ordered_numbers(lines)

    if len(numbers) == 3:

        profile["Posts"], profile["Followers"], profile["Following"] = numbers

    desc_lines = []

    for i, line in enumerate(lines):

        clean_line = line.strip()

        if not profile["UserName"] and re.match(r'^[\w.]+$', clean_line) and i < 5:

            profile["UserName"] = clean_line

            profile["UserName_LetterCount"] = len(re.sub(r'\W+', '', clean_line))

            profile["UserName_DigitCount"] = sum(char.isdigit() for char in clean_line)

        if not profile["FullName"] and len(clean_line.split()) >= 2 and not any(char.isdigit() for char in clean_line):

            if "DVM" in clean_line or "Dr" in clean_line or clean_line.istitle():

                profile["FullName"] = clean_line

                profile["FullName_LetterCount"] = len(re.sub(r'[^A-Za-z]', '', clean_line))

                profile["FullName_DigitCount"] = sum(char.isdigit() for char in clean_line)

                profile["FullName_WordCount"] = len(clean_line.split())

        if any(word in clean_line.lower() for word in ["veterinarian", "certified", "#", "follow", "voice", "honest"]):

            desc_lines.append(clean_line)

    profile["description"] = " ".join(desc_lines).strip()

    if profile["UserName"] and profile["FullName"]:

        uname = re.sub(r'\W+', '', profile["UserName"].lower())

        fname = re.sub(r'\W+', '', profile["FullName"].lower())

        similarity = SequenceMatcher(None, uname, fname).ratio()

        profile["Match"] = 1 if similarity > 0.5 else 0

    if profile["UserName_LetterCount"] > 0:

        profile["nums/length username"] = round(profile["UserName_DigitCount"] / profile["UserName_LetterCount"], 3)

    if profile["FullName_LetterCount"] > 0:

        profile["nums/length fullname"] = round(profile["FullName_DigitCount"] / profile["FullName_LetterCount"], 3)

    return profile


def figure_to_base64():
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_feature_order():
    return [
        "#followers",
        "#follows",
        "#posts",
        "following_follower_ratio",
        "posts_per_follower",
        "activity_score",
        "engagement_ratio",
    ]


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/graphs")
def graphs():
    plot_url = None
    try:
        last_prediction = session.get("last_prediction")

        if last_prediction and "features" in last_prediction:
            feature_order = get_feature_order()
            feature_values = [last_prediction["features"].get(name, 0.0) for name in feature_order]

            plt.figure(figsize=(10, 4.8))
            sns.barplot(x=feature_order, y=feature_values, hue=feature_order, dodge=False, palette="Set2", legend=False)
            plt.title("Current Image Feature Values")
            plt.xlabel("Feature")
            plt.ylabel("Value")
            plt.xticks(rotation=20, ha="right")
            plot_url = figure_to_base64()
    except Exception as e:
        logger.warning(f"Could not build graphs page: {e}")

    return render_template("graphs.html", plot_url=plot_url)


@app.route("/shap")
def shap_page():
    shap_url = None
    try:
        last_prediction = session.get("last_prediction")
        if last_prediction and "features" in last_prediction:
            feature_names = get_feature_order()
            x_values = np.array([last_prediction["features"].get(name, 0.0) for name in feature_names], dtype=float)

            rf_est = None
            for est in getattr(clf, "estimators_", []):
                if hasattr(est, "feature_importances_"):
                    rf_est = est
                    break

            if rf_est is not None and hasattr(scaler_x, "mean_") and hasattr(scaler_x, "scale_"):
                means = np.array(scaler_x.mean_, dtype=float)
                scales = np.array(scaler_x.scale_, dtype=float)
                scales = np.where(scales == 0, 1.0, scales)
                z_scores = (x_values - means) / scales
                impact = rf_est.feature_importances_ * z_scores

                pairs = sorted(zip(feature_names, impact), key=lambda x: abs(x[1]), reverse=True)
                names = [p[0] for p in pairs]
                values = [p[1] for p in pairs]
                colors = ["#ef4444" if v > 0 else "#10b981" for v in values]

                plt.figure(figsize=(9, 4.8))
                plt.barh(names, values, color=colors)
                plt.title("Per-Image Feature Impact (SHAP-style)")
                plt.xlabel("Impact on Prediction")
                plt.ylabel("Feature")
                plt.axvline(x=0, color="white", linewidth=1, alpha=0.5)
                plt.gca().invert_yaxis()
                shap_url = figure_to_base64()
    except Exception as e:
        logger.warning(f"Could not build shap page: {e}")

    return render_template("shap.html", shap_url=shap_url)


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        file = request.files["image"]

        image = Image.open(file.stream)

        face = check_profile_face(image)

        lines = extract_text_from_image(image)

        info = parse_profile_data(lines)

        followers = max(1, info.get("Followers") or 1)

        follows = max(1, info.get("Following") or 1)

        posts = max(1, info.get("Posts") or 1)

        following_follower_ratio = follows / followers

        posts_per_follower = posts / followers

        activity_score = posts / follows

        engagement_ratio = followers / posts


        features = {

            "#followers": followers,
            "#follows": follows,
            "#posts": posts,

            "following_follower_ratio": following_follower_ratio,
            "posts_per_follower": posts_per_follower,
            "activity_score": activity_score,
            "engagement_ratio": engagement_ratio

        }

        df = pd.DataFrame([features])

        X_scaled = scaler_x.transform(df)

        pred = clf.predict(X_scaled)[0]

        pred_proba = clf.predict_proba(X_scaled)[0]

        confidence_score = max(pred_proba)


        # ======================================
        # 🔥 LABELS REVERSED HERE
        # ======================================

        result = {

            "prediction": "Fake" if pred == 0 else "Real",

            "confidence": f"{confidence_score * 100:.1f}%",

            "features": {

                "Profile Picture": "Has Face" if face == 1 else "No Face",

                "Followers": followers,

                "Following": follows,

                "Posts": posts

            }

        }

        # Store latest prediction so /graphs and /shap are per uploaded image.
        session["last_prediction"] = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "features": {k: float(v) for k, v in features.items()},
        }

        return render_template("predict.html", result=result)

    return render_template("predict.html", result=None)



# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    app.run(debug=True)