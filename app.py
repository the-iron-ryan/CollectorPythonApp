import io

from flask import Flask, jsonify, request
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

# Load the classifier once when the app starts
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
plant_classifier = pipeline("image-classification", model="umutbozdag/plant-identity")


@app.route("/")
def index():
    return "Hello!"


@app.route("/classify_image", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    pil_image = Image.open(io.BytesIO(image_bytes))

    results = plant_classifier(pil_image)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
