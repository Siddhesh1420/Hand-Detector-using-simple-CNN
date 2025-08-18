import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

H,W = 350,500
Z=1220

def load_data(image_dir):
    X = []
    y = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not load image: {img_path}")
            continue  # skip this image
        h,w, _ = img.shape
        img = img[h-H:h,w-W:w] / 255.0

        X.append(img)
        t=(1 if int(filename.split('_')[-1].split('.')[0]) in range(Z) else 0)  # Assuming 'hand' indicates positive samples
        y.append(t)  # with probability 1

    return np.array(X), np.array(y)
X, y = load_data("Dataset")
models= models.Sequential([
    layers.Input(shape=(H, W, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1,activation='sigmoid')  # 1 for probability of hand presence
])
models.compile(optimizer='adam', loss="binary_crossentropy")
models.fit(X, y, epochs=10, batch_size=8, validation_split=0.2)
models.save("hand_detector.h5")