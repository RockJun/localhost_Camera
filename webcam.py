import cv2
import os
import time
import threading
from datetime import datetime
import face_recognition

# 配置部分
CONFIG = {
    'photo_dir': 'D:\\Photos',
    'camera_index': 0,
    'frame_width': 320,
    'frame_height': 240,
    'resize_ratio': 0.25,  # 缩放比例
    'capture_interval': 60  # 捕获间隔时间（秒）
}

# 创建存储照片的目录
if not os.path.exists(CONFIG['photo_dir']):
    os.makedirs(CONFIG['photo_dir'])

# 初始化摄像头
try:
    cap = cv2.VideoCapture(CONFIG['camera_index'])
    cap.set(3, CONFIG['frame_width'])  # 设置分辨率的宽
    cap.set(4, CONFIG['frame_height'])  # 设置分辨率的高
except Exception as e:
    print("Error initializing camera: {}".format(e))
    exit(1)

last_capture_time = None
reference_face = None
frame_lock = threading.Lock()
process_this_frame = True
shared_frame = None

def save_face_picture(current_face, now):
    """保存当前帧中检测到的人脸。"""
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    photo_name = "{}.png".format(timestamp)
    photo_path = os.path.join(CONFIG['photo_dir'], photo_name)
    cv2.imwrite(photo_path, current_face)
    print("Photo taken and saved to {}".format(photo_path))

def face_recognition_handler():
    """处理面部识别的线程函数。"""
    global process_this_frame, reference_face, last_capture_time, frame_lock, shared_frame
    
    while True:
        with frame_lock:
            if not process_this_frame or shared_frame is None:
                continue

            process_this_frame = False

            small_frame = cv2.resize(shared_frame, (0, 0), fx=CONFIG['resize_ratio'], fy=CONFIG['resize_ratio'])
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            now = datetime.now()

            if face_locations:
                # 检测到的新面部
                new_face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]
                # 如果是新面孔或者距上一次捕获超过配置时间
                if reference_face is None or not face_recognition.compare_faces([reference_face], new_face_encoding)[0] or (last_capture_time and (now - last_capture_time).seconds >= CONFIG['capture_interval']):
                    save_face_picture(shared_frame, now)
                    reference_face = new_face_encoding
                    last_capture_time = now

threading.Thread(target=face_recognition_handler, daemon=True).start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with frame_lock:
            shared_frame = frame.copy()
            process_this_frame = True
        
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print("An error occurred: {}".format(e))
finally:
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
