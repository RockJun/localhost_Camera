import cv2
import os
import time
from datetime import datetime
import face_recognition
import threading

photo_dir = 'D:\\Photos'
if not os.path.exists(photo_dir):
    os.makedirs(photo_dir)

cap = cv2.VideoCapture(0)

cap.set(3, 320)  # 设置分辨率的宽
cap.set(4, 240)  # 设置分辨率的高

last_capture_time = None
reference_face = None
frame_lock = threading.Lock()
process_this_frame = True
shared_frame = None

def save_face_picture(current_face, now):
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    photo_name = "{}.png".format(timestamp)
    photo_path = os.path.join(photo_dir, photo_name)
    cv2.imwrite(photo_path, current_face)
    print("Photo taken and saved to {}".format(photo_path))

def face_recognition_handler():
    global process_this_frame, reference_face, last_capture_time, frame_lock, shared_frame  # 确保 shared_frame 在这里声明
    
    while True:
        with frame_lock:
            if not process_this_frame or shared_frame is None:
                continue

            process_this_frame = False

            # 使用 shared_frame
            small_frame = cv2.resize(shared_frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            now = datetime.now()

            if face_locations:
                new_face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]
                if reference_face is None or not face_recognition.compare_faces([reference_face], new_face_encoding)[0] or (last_capture_time and (now - last_capture_time).seconds >= 60):
                    save_face_picture(shared_frame, now)
                    reference_face = new_face_encoding
                    last_capture_time = now

# 启动面部识别线程
threading.Thread(target=face_recognition_handler, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    with frame_lock:
        shared_frame = frame.copy()  # 将当前帧复制到 shared_frame
        process_this_frame = True
    
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
