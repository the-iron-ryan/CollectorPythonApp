from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the classifier once when the app starts
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

@app.route("/classify_image", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    results = classifier(image_bytes)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
