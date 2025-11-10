from flask import Flask, render_template, send_from_directory, jsonify
import os
import json

app = Flask(__name__)

ALERTS_FILE = "alerts/alerts.json"
ALERTS_DIR = "alerts"

@app.route("/alerts/<path:filename>")
def serve_alert_image(filename):
    return send_from_directory(ALERTS_DIR, filename)

@app.route("/api/alerts")
def get_alerts():
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
    else:
        alerts = []
    return jsonify(alerts)

@app.route("/api/alerts/images")
def get_image_alerts():
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
            image_alerts = [a for a in alerts if a.get("type") == "image"]
    else:
        image_alerts = []
    return jsonify(image_alerts)

@app.route("/api/alerts/texts")
def get_text_alerts():
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
            text_alerts = [a for a in alerts if a.get("type") == "text"]
    else:
        text_alerts = []
    return jsonify(text_alerts)

@app.route("/")
def index():
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
    else:
        alerts = []
    
    image_alerts = [a for a in alerts if a.get("type") == "image"]
    text_alerts = [a for a in alerts if a.get("type") == "text"]
    
    return render_template("alerts.html", 
                          image_alerts=image_alerts, 
                          text_alerts=text_alerts,
                          all_alerts=alerts)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)