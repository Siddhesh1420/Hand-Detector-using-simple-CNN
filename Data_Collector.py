import cv2
import numpy as np

IMG_SIZE = 224

cap = cv2.VideoCapture(0)
save = False
i=0
H,W = 350,500
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    input_img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) / 255.0

    h, w, _ = frame.shape

    x1, y1 = h-H, w-W
    x2, y2 = h,w
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f"Hand please", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        save = not save
    if save:
        filename = f"Dataset/saved_hand_{i}.jpg"
        i+=1
        cv2.imwrite(filename, frame)
        print('Saved', filename)

cap.release()
cv2.destroyAllWindows()
