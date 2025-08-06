import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("hand_detector.h5")
IMG_SIZE = 224

cap = cv2.VideoCapture(0)
H,W = 350,500

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    frame = cv2.flip(frame, 1)
    input_data = frame[h-H:h,w-W:w]

    input_img = input_data / 255.0
    pred = model.predict(np.expand_dims(input_img, axis=0))[0]

    h,w = frame.shape[:2]

    x1, y1 = h-H, w-W
    x2, y2 = h,w
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f"Hand: {float(pred):.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
