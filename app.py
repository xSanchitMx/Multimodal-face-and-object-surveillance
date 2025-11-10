from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import time
import json

app = Flask(__name__)

UPLOAD_FOLDER = "data/new_faces"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
QUERIES_FILE = "data/new_queries.txt"
IMAGE_INDEX_PATH = "data/image_paths.txt"
TEXT_INDEX_PATH = "data/text_paths.txt"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "text_query" in request.form and request.form["text_query"].strip():
            text_query = request.form["text_query"].strip()
            with open(QUERIES_FILE, "a") as f:
                f.write(f"txt:{text_query}\n")
            return redirect(url_for("index"))

        if "image_query" in request.files:
            img = request.files["image_query"]
            if img.filename != "":
                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}_{img.filename}"
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                img.save(save_path)

                with open(QUERIES_FILE, "a") as f:
                    f.write(f"img:{save_path}\n")
            return redirect(url_for("index"))

    return render_template("index.html")

@app.route("/api/queries")
def get_queries():
    image_queries = []
    text_queries = []
    
    if os.path.exists(IMAGE_INDEX_PATH):
        with open(IMAGE_INDEX_PATH, "r") as f:
            image_queries = [os.path.basename(p) for p in f.read().splitlines()]
    
    if os.path.exists(TEXT_INDEX_PATH):
        with open(TEXT_INDEX_PATH, "r") as f:
            text_queries = f.read().splitlines()
    
    return jsonify({
        "image_queries": image_queries,
        "text_queries": text_queries
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)