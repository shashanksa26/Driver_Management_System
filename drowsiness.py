import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
import os
from threading import Thread
import psutil
import base64
from datetime import datetime
from MongoDB import employees, db

# Global variables and constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 100
YAWN_THRESH = 25
HEAD_POSE_THRESH = 18
ALARM_COOLDOWN = 3
ALERT_DISPLAY_TIME = 3.0  # seconds after last detection
RESIZE_FACTOR = 0.5
FONT_SCALE = 0.7
FONT_THICKNESS = 2
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
NO_FACE_MAX = 60  # Number of frames with no face before returning to recognition

# Initialize pygame mixer for sound
pygame.mixer.init()

# Add alerts collection
alerts = db['alerts']

# Helper functions

def get_scaled_font_scale(scale_factor=1.0):
    return FONT_SCALE * scale_factor

def get_scaled_thickness(scale_factor=1.0):
    return int(FONT_THICKNESS * scale_factor)

def calculate_head_pose(landmarks):
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    eye_center = np.mean([left_eye, right_eye], axis=0)
    mouth_center = np.mean([left_mouth, right_mouth], axis=0)
    dx = mouth_center[0] - eye_center[0]
    dy = mouth_center[1] - eye_center[1]
    angle = np.degrees(np.arctan2(dx, dy))
    return angle

def sound_alarm(path):
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing sound: {e}")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(landmarks):
    left_eye = np.array([landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144]])
    right_eye = np.array([landmarks[362], landmarks[385], landmarks[387], landmarks[263], landmarks[373], landmarks[380]])
    leftEAR = eye_aspect_ratio(left_eye)
    rightEAR = eye_aspect_ratio(right_eye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, left_eye, right_eye)

def lip_distance(landmarks):
    top_lip = np.array([landmarks[61], landmarks[291], landmarks[0]])
    low_lip = np.array([landmarks[291], landmarks[405], landmarks[17]])
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def save_alert(employee_id, name, alert_type, frame):
    # Encode frame as base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    alert_doc = {
        'employee_id': employee_id,
        'name': name,
        'alert_type': alert_type,
        'timestamp': datetime.now(),
        'frame_base64': img_base64
    }
    alerts.insert_one(alert_doc)

def run_drowsiness_monitor(user_id, webcam_index=0, alarm_path="Alert.WAV"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )
    print(f"-> Starting Drowsiness Monitor for user: {user_id}")
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from webcam index {webcam_index}.")
        return
    cv2.namedWindow("Driver Monitoring", cv2.WINDOW_NORMAL)
    COUNTER = 0
    last_alarm_time = 0
    last_alert_time = 0
    current_alert = ""
    prev_frame_time = 0
    memory_usage = 0
    no_face_counter = 0
    # Parse user_id for name and id
    if '(' in user_id and user_id.endswith(')'):
        name, emp_id = user_id.rsplit('(', 1)
        name = name.strip()
        emp_id = emp_id[:-1].strip()
    else:
        name = user_id
        emp_id = user_id
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame. Exiting...")
            break
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time else 0
        prev_frame_time = new_frame_time
        fps = int(fps)
        scale_factor = min(WINDOW_WIDTH / frame.shape[1], WINDOW_HEIGHT / frame.shape[0])
        new_width = int(frame.shape[1] * scale_factor)
        new_height = int(frame.shape[0] * scale_factor)
        frame = cv2.resize(frame, (new_width, new_height))
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (int(width * RESIZE_FACTOR), int(height * RESIZE_FACTOR)))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        ear = 0
        distance = 0
        head_angle = 0
        if results.multi_face_landmarks:
            no_face_counter = 0
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
                head_angle = calculate_head_pose(landmarks)
                eye = final_ear(landmarks)
                ear = eye[0]
                distance = lip_distance(landmarks)
                # Drowsiness detection
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        current_time = time.time()
                        if (current_time - last_alarm_time) > ALARM_COOLDOWN:
                            last_alarm_time = current_time
                            current_alert = "DROWSINESS ALERT!"
                            last_alert_time = current_time
                            t = Thread(target=sound_alarm, args=(alarm_path,))
                            t.daemon = True
                            t.start()
                            save_alert(emp_id, name, current_alert, frame)
                else:
                    COUNTER = 0
                # Yawn detection
                if distance > YAWN_THRESH:
                    current_time = time.time()
                    if (current_time - last_alarm_time) > ALARM_COOLDOWN:
                        last_alarm_time = current_time
                        current_alert = "YAWN ALERT!"
                        last_alert_time = current_time
                        t = Thread(target=sound_alarm, args=(alarm_path,))
                        t.daemon = True
                        t.start()
                        save_alert(emp_id, name, current_alert, frame)
                # Distraction detection
                if abs(head_angle) > HEAD_POSE_THRESH:
                    current_time = time.time()
                    if (current_time - last_alarm_time) > ALARM_COOLDOWN:
                        last_alarm_time = current_time
                        current_alert = "DISTRACTION ALERT!"
                        last_alert_time = current_time
                        t = Thread(target=sound_alarm, args=(alarm_path,))
                        t.daemon = True
                        t.start()
                        save_alert(emp_id, name, current_alert, frame)
        else:
            no_face_counter += 1
        # Display metrics
        text_y_pos = int(30 * scale_factor)
        text_x_pos = int(frame.shape[1] - 200)
        cv2.putText(frame, f"User: {user_id}", (10, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, get_scaled_font_scale(scale_factor), (255, 255, 0), get_scaled_thickness(scale_factor))
        cv2.putText(frame, f"EAR: {ear:.2f}", (text_x_pos, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, get_scaled_font_scale(scale_factor), (0, 0, 255), get_scaled_thickness(scale_factor))
        cv2.putText(frame, f"YAWN: {distance:.2f}", (text_x_pos, text_y_pos + int(30 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, get_scaled_font_scale(scale_factor), (0, 0, 255), get_scaled_thickness(scale_factor))
        cv2.putText(frame, f"HEAD ANGLE: {head_angle:.1f}°", (text_x_pos, text_y_pos + int(60 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, get_scaled_font_scale(scale_factor), (0, 0, 255), get_scaled_thickness(scale_factor))
        cv2.putText(frame, f"FPS: {fps}", (text_x_pos, text_y_pos + int(90 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, get_scaled_font_scale(scale_factor), (0, 0, 255), get_scaled_thickness(scale_factor))
        memory_usage = get_memory_usage()
        cv2.putText(frame, f"MEM: {memory_usage:.1f}MB", (text_x_pos, text_y_pos + int(120 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, get_scaled_font_scale(scale_factor), (0, 0, 255), get_scaled_thickness(scale_factor))
        # Display alert text if within display time
        current_time = time.time()
        if current_time - last_alert_time < ALERT_DISPLAY_TIME and current_alert:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            text_size = cv2.getTextSize(current_alert, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] - 50
            cv2.putText(frame, current_alert, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
        cv2.imshow("Driver Monitoring", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if no_face_counter >= NO_FACE_MAX:
            print("No face detected for a while, returning to recognition.")
            break
    cap.release()
    cv2.destroyAllWindows() 