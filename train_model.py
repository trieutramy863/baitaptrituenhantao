import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================
# CẤU HÌNH
# =====================
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
DATASET_DIR = "dataset"

# =====================
# DATA AUGMENTATION
# =====================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False   # QUAN TRỌNG để vẽ confusion matrix
)

# =====================
# MOBILENETV2 MODEL
# =====================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================
# TRAIN
# =====================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# =====================
# VẼ LEARNING CURVE
# =====================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Learning Curve - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curve - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("learning_curve.png")  # lưu ảnh
plt.show()

# =====================
# CONFUSION MATRIX
# =====================
# Dự đoán
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Nhãn thật
y_true = val_generator.classes

# Tính ma trận
cm = confusion_matrix(y_true, y_pred_classes)

# Lấy tên class
class_names = list(val_generator.class_indices.keys())

# Vẽ
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.savefig("confusion_matrix.png")  # lưu ảnh
plt.show()

# =====================
# SAVE MODEL
# =====================
model.save("model.h5")
print("✅ Đã lưu model.h5")