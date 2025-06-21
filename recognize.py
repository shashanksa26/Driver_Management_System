import cv2
import numpy as np
from MongoDB import employees
from face_model import extract_embedding
from face_utils import cosine_similarity
from drowsiness import run_drowsiness_monitor

def recognize_faces(face_detection):
    while True:
        employee_data = list(employees.find({}))
        known_embeddings = {emp['employee_id']: np.array(emp['avg_embedding'], dtype=np.float32) for emp in employee_data}
        if not known_embeddings:
            print("No employees registered in database!")
            return
        cap = cv2.VideoCapture(0)
        recognition_active = True
        recognition_threshold = 0.65
        NO_FACE_MAX = 60  # ~2 seconds at 30 FPS
        no_face_counter = 0
        cv2.namedWindow("Driver Monitoring", cv2.WINDOW_NORMAL)
        user_verified = False
        while recognition_active:
            ret, frame = cap.read()
            if not ret:
                continue
            # Downscale for recognition to reduce memory and match model input
            small_frame = cv2.resize(frame, (160, 160))
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            detected = False
            if results.detections:
                for detection in results.detections:
                    detected = True
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    x = max(0, x - 20)
                    y = max(0, y - 30)
                    w = min(iw - x, w + 40)
                    h = min(ih - y, h + 60)
                    face_img = frame[y:y+h, x:x+w]
                    embedding = extract_embedding(face_img)
                    best_match = None
                    best_similarity = 0
                    for emp_id, ref_embedding in known_embeddings.items():
                        if embedding.shape != ref_embedding.shape:
                            print(f"[WARN] Shape mismatch: {embedding.shape} vs {ref_embedding.shape} for {emp_id}. Skipping.")
                            continue
                        similarity = cosine_similarity(embedding, ref_embedding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = emp_id
                    color = (0, 255, 0) if best_similarity > recognition_threshold else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    if best_similarity > recognition_threshold:
                        emp = employees.find_one({"employee_id": best_match})
                        name = emp['name']
                        cv2.putText(frame, f"{name} ({best_match})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Confidence: {best_similarity:.2f}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                        cv2.imshow("Driver Monitoring", frame)
                        cv2.waitKey(1000)
                        cap.release()
                        cv2.destroyAllWindows()
                        run_drowsiness_monitor(f"{name} ({best_match})", webcam_index=0, alarm_path="Alert.WAV")
                        user_verified = True
                        recognition_active = False
                        break
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if detected:
                    no_face_counter = 0
            else:
                no_face_counter += 1
            cv2.imshow("Driver Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                recognition_active = False
                user_verified = False
            if no_face_counter >= NO_FACE_MAX:
                # No face detected for a while, just keep waiting for a new face
                no_face_counter = 0
        cap.release()
        cv2.destroyAllWindows()
        if not user_verified:
            break  # User quit with 'q', exit recognition
        # If user_verified, loop again to recognize next user after drowsiness monitor ends 