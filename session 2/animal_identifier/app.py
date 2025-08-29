from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load MobileNetV2 model
model = MobileNetV2(weights="imagenet")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save upload temporarily
    img_path = os.path.join("static", "uploads", file.filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    file.save(img_path)

    # Preprocess for MobileNet
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]  # top-3 predictions

    predictions = [{"label": lbl, "description": desc, "probability": float(prob)} for (lbl, desc, prob) in decoded]

    return jsonify({
        "filename": file.filename,
        "predictions": predictions
    })

if __name__ == "__main__":
    app.run(debug=True)
