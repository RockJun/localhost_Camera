import cv2
import os
import time
from datetime import datetime
import face_recognition

photo_dir = 'D:\\Photos'
if not os.path.exists(photo_dir):
    os.makedirs(photo_dir)

cap = cv2.VideoCapture(0)

last_capture_time = None
reference_face = None

def save_face_picture(current_face, now):
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    photo_name = f"{timestamp}.png"
    photo_path = os.path.join(photo_dir, photo_name)
    cv2.imwrite(photo_path, current_face)
    print(f"Photo taken and saved to {photo_path}")

TARGET_WIDTH = 320

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    scale_factor = TARGET_WIDTH / frame.shape[1]
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    now = datetime.now()

    if face_locations:
        new_face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]

        if reference_face is None:
            save_face_picture(frame, now)  # 保存当前帧
            reference_face = new_face_encoding
            last_capture_time = now

        elif not face_recognition.compare_faces([reference_face], new_face_encoding)[0] or (last_capture_time and (now - last_capture_time).seconds >= 60):
            save_face_picture(frame, now)  # 保存当前帧
            reference_face = new_face_encoding
            last_capture_time = now

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
