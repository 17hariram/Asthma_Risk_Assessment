import os
import json
import datetime

from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import joblib

# ----------------- BASIC SETUP -----------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATIENT_FILE = os.path.join(BASE_DIR, "patient.json")
READINGS_FILE = os.path.join(BASE_DIR, "readings.json")
ATTACKS_FILE = os.path.join(BASE_DIR, "attacks.json")
ALERTS_FILE = os.path.join(BASE_DIR, "alerts.json")

MODEL_FUTURE_FILE = os.path.join(BASE_DIR, "xgb_future.pkl")
FEATURES_FUTURE_FILE = os.path.join(BASE_DIR, "features_future.json")

future_model = None
FEATURES_FUTURE = []

# ----------------- UTILITIES -----------------


def default_patient():
    """Default patient profile (Hari, 21, etc.)."""
    return {
        "name": "Hari",
        "age": 21,
        "gender": "Male",
        "smoker": "Non-smoker",          # Non-smoker / Passive smoker / Active smoker
        "allergy_present": "No",         # Yes / No
        "allergy_type": "None",          # None / Dust / Pollen / Pets / Other
        "occupation": "Home/Office"      # Home/Office / Outdoor/Traffic / Factory/Heavy
    }


def load_json(path, default=None):
    if default is None:
        default = {}
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving {path}: {e}")


# ----------------- LOAD FUTURE MODEL -----------------

try:
    future_model = joblib.load(MODEL_FUTURE_FILE)
    with open(FEATURES_FUTURE_FILE, "r") as f:
        FEATURES_FUTURE = json.load(f)
    print("[INFO] Loaded future model and features.")
except Exception as e:
    print("[WARN] Could not load future model:", e)
    future_model = None
    FEATURES_FUTURE = []


# ----------------- ML INFERENCE -----------------


def run_future_model(sensor, patient):
    """
    sensor: dict with humidity, temperature, mq2, mq135, dust or dust_equiv
    patient: dict with profile from patient.json

    Model trained with features:
    [humidity, temperature, mq2, mq135, dust,
     age, smoker_level, allergy_present, allergy_type, occupation_risk]
    """
    if future_model is None or not FEATURES_FUTURE:
        return None, "UNKNOWN"

    # --- Sensor features ---
    hum = float(sensor.get("humidity", 0))
    temp = float(sensor.get("temperature", 0))
    mq2 = float(sensor.get("mq2", 0))
    mq135 = float(sensor.get("mq135", 0))
    dust = float(sensor.get("dust_equiv", sensor.get("dust", 0)))

    # --- Patient features ---
    patient = patient or default_patient()

    age = float(patient.get("age", 21))

    # smoker_level: 0=Non, 1=Passive, 2=Active
    smoker_str = patient.get("smoker", "Non-smoker")
    smoker_map = {
        "Non-smoker": 0,
        "Passive smoker": 1,
        "Active smoker": 2
    }
    smoker_level = float(smoker_map.get(smoker_str, 0))

    # allergy_present: Yes/No
    allergy_present_str = patient.get("allergy_present", "No")
    allergy_present = 1.0 if allergy_present_str == "Yes" else 0.0

    # allergy_type: None/Dust/Pollen/Pets/Other
    allergy_type_str = patient.get("allergy_type", "None")
    allergy_type_map = {
        "None": 0,
        "Dust": 1,
        "Pollen": 2,
        "Pets": 3,
        "Other": 4
    }
    allergy_type = float(allergy_type_map.get(allergy_type_str, 0))

    # occupation_risk: 0=Home/Office, 1=Outdoor/Traffic, 2=Factory/Heavy
    occupation_str = patient.get("occupation", "Home/Office")
    occ_map = {
        "Home/Office": 0,
        "Outdoor/Traffic": 1,
        "Factory/Heavy": 2
    }
    occupation_risk = float(occ_map.get(occupation_str, 0))

    feat_map = {
        "humidity": hum,
        "temperature": temp,
        "mq2": mq2,
        "mq135": mq135,
        "dust": dust,
        "age": age,
        "smoker_level": smoker_level,
        "allergy_present": allergy_present,
        "allergy_type": allergy_type,
        "occupation_risk": occupation_risk
    }

    vec = [float(feat_map.get(f, 0.0)) for f in FEATURES_FUTURE]
    X = np.array([vec])

    prob = float(future_model.predict_proba(X)[0][1])

    if prob < 0.35:
        label = "LOW RISK"
    elif prob < 0.65:
        label = "MEDIUM RISK"
    else:
        label = "HIGH RISK"

    return prob, label


# ----------------- API ENDPOINTS -----------------


@app.route("/api/sensor", methods=["POST"])
def api_sensor():
    """
    ESP8266 posts JSON here:
    {
      "temperature": 30.2,
      "humidity": 60,
      "mq2": 1,
      "mq135": 1,
      "dust": 95   // or "dust_equiv"
    }
    """
    data = request.get_json(force=True) or {}

    temp = float(data.get("temperature", 0))
    hum = float(data.get("humidity", 0))
    mq2 = int(data.get("mq2", 1))
    mq135 = int(data.get("mq135", 1))
    dust = float(data.get("dust_equiv", data.get("dust", 0)))

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    # Store reading
    reading = {
        "timestamp": timestamp,
        "temperature": temp,
        "humidity": hum,
        "mq2": mq2,
        "mq135": mq135,
        "dust_equiv": dust
    }

    readings = load_json(READINGS_FILE, [])
    readings.append(reading)
    save_json(READINGS_FILE, readings)

    # Load patient & run model
    patient = load_json(PATIENT_FILE, default_patient())
    sensor_dict = {
        "temperature": temp,
        "humidity": hum,
        "mq2": mq2,
        "mq135": mq135,
        "dust_equiv": dust
    }
    prob, label = run_future_model(sensor_dict, patient)

    # Log only HIGH RISK as attack + alert
    if prob is not None and "HIGH" in label:
        attacks = load_json(ATTACKS_FILE, [])
        attacks.append({
            "timestamp": timestamp,
            "probability": prob,
            "risk_label": label,
            "details": f"T={temp}, H={hum}, MQ2={mq2}, MQ135={mq135}, Dust={dust}"
        })
        save_json(ATTACKS_FILE, attacks)

        alerts = load_json(ALERTS_FILE, [])
        alerts.append({
            "timestamp": timestamp,
            "message": "HIGH Immediate Asthma Risk Detected!",
            "details": f"T={temp}, H={hum}, MQ2={mq2}, MQ135={mq135}, Dust={dust}"
        })
        save_json(ALERTS_FILE, alerts)

    return jsonify({
        "status": "ok",
        "timestamp": timestamp,
        "probability": prob,
        "label": label
    })


# optional simple test endpoint
@app.route("/predict", methods=["POST"])
def predict_once():
    data = request.get_json(force=True) or {}

    temp = float(data.get("temperature", 0))
    hum = float(data.get("humidity", 0))
    mq2 = int(data.get("mq2", 1))
    mq135 = int(data.get("mq135", 1))
    dust = float(data.get("dust_equiv", data.get("dust", 0)))

    patient = load_json(PATIENT_FILE, default_patient())
    sensor_dict = {
        "temperature": temp,
        "humidity": hum,
        "mq2": mq2,
        "mq135": mq135,
        "dust_equiv": dust
    }
    prob, label = run_future_model(sensor_dict, patient)

    return jsonify({"probability": prob, "label": label})


# ----------------- WEB PAGES -----------------


@app.route("/")
@app.route("/dashboard")
def dashboard():
    readings = load_json(READINGS_FILE, [])
    latest = readings[-1] if readings else None
    patient = load_json(PATIENT_FILE, default_patient())

    latest_attack = None
    if latest and future_model is not None:
        sensor_dict = {
            "temperature": latest.get("temperature", 0),
            "humidity": latest.get("humidity", 0),
            "mq2": latest.get("mq2", 0),
            "mq135": latest.get("mq135", 0),
            "dust_equiv": latest.get("dust_equiv", latest.get("dust", 0))
        }
        prob, label = run_future_model(sensor_dict, patient)
        if prob is not None:
            latest_attack = {
                "timestamp": latest.get("timestamp", ""),
                "probability": prob,
                "risk_label": label,
                "details": f"T={sensor_dict['temperature']}, "
                           f"H={sensor_dict['humidity']}, "
                           f"MQ2={sensor_dict['mq2']}, "
                           f"MQ135={sensor_dict['mq135']}, "
                           f"Dust={sensor_dict['dust_equiv']}"
            }

    return render_template(
        "dashboard.html",
        latest=latest,
        latest_attack=latest_attack,
        patient=patient
    )


@app.route("/patient_profile")
def patient_profile():
    patient = load_json(PATIENT_FILE, default_patient())
    return render_template("profile.html", patient=patient)


@app.route("/update_profile", methods=["GET", "POST"])
def update_profile():
    patient = load_json(PATIENT_FILE, default_patient())

    if request.method == "POST":
        patient["name"] = request.form.get("name", patient.get("name", "Hari"))
        try:
            patient["age"] = int(request.form.get("age", patient.get("age", 21)))
        except ValueError:
            patient["age"] = patient.get("age", 21)

        patient["gender"] = request.form.get("gender", patient.get("gender", "Male"))
        patient["smoker"] = request.form.get("smoker", patient.get("smoker", "Non-smoker"))
        patient["allergy_present"] = request.form.get(
            "allergy_present", patient.get("allergy_present", "No")
        )
        patient["allergy_type"] = request.form.get(
            "allergy_type", patient.get("allergy_type", "None")
        )
        patient["occupation"] = request.form.get(
            "occupation", patient.get("occupation", "Home/Office")
        )

        save_json(PATIENT_FILE, patient)
        return redirect(url_for("patient_profile"))

    return render_template("update_profile.html", patient=patient)


@app.route("/alerts")
def alerts_page():
    alerts = load_json(ALERTS_FILE, [])
    return render_template("alerts.html", alerts=alerts)


@app.route("/attack_history")
def attack_history():
    attacks = load_json(ATTACKS_FILE, [])
    # newest first
    attacks = list(reversed(attacks))
    return render_template("attack_history.html", attacks=attacks)


@app.route("/graphs")
def graphs():
    readings = load_json(READINGS_FILE, [])

    # keep last 300 readings for clarity
    if len(readings) > 300:
        readings = readings[-300:]

    for r in readings:
        ts = r.get("timestamp", "")
        try:
            dt = datetime.datetime.fromisoformat(str(ts))
            r["time"] = dt.strftime("%H:%M:%S")
        except Exception:
            r["time"] = str(ts)

        if "dust_equiv" not in r:
            r["dust_equiv"] = r.get("dust", 0)

    return render_template("graphs.html", readings=readings)


# ----------------- RUN SERVER -----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
