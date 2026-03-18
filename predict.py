import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model đã train
model = tf.keras.models.load_model("model.h5")

# Thứ tự nhãn PHẢI trùng lúc train & app.py
labels = ['huuco', 'nguyhai', 'taiche']

def predict_image(image_path):
    """
    Dự đoán loại rác từ ảnh
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Không đọc được ảnh")

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize đúng MobileNetV2
    img = cv2.resize(img, (224, 224))

    # Thêm batch dimension
    img = np.expand_dims(img, axis=0)

    # Preprocess đúng chuẩn MobileNetV2
    img = preprocess_input(img)

    preds = model.predict(img)
    class_index = np.argmax(preds)
    confidence = float(preds[0][class_index]) * 100

    return labels[class_index], round(confidence, 2)


# Test nhanh khi chạy trực tiếp file
if __name__ == "__main__":
    label, conf = predict_image("dataset/huuco/1.jpg")
    print(label, conf)
