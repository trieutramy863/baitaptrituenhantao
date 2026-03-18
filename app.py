from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")
labels = ['huuco', 'nguyhai', 'taiche']

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join("static/upload", file.filename)
            file.save(image_path)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            preds = model.predict(img)
            class_index = np.argmax(preds)
            result = labels[class_index]
            confidence = round(float(preds[0][class_index]) * 100, 2)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
