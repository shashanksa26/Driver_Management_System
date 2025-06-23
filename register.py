import cv2
import numpy as np
from datetime import datetime
from MongoDB import employees
from face_model import extract_embedding

def register_new_employee(face_detection):
    employee_id = input("Enter employee ID: ")
    name = input("Enter employee name: ")
    
    if employees.find_one({"employee_id": employee_id}):
        print(f"Employee {employee_id} already exists!")
        return
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 160)
    embeddings = []
    sample_count = 0
    required_samples = 5
    print(f"Press SPACE to start capturing {required_samples} face samples, or 'q' to quit.")

    # Wait for space bar to start
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, "Press SPACE to start capturing, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Registration - Waiting to Start", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Registration cancelled by user.")
            return

    print(f"Capturing {required_samples} face samples...")
    while sample_count < required_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
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
                embeddings.append(embedding.tolist())
                sample_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Sample: {sample_count}/{required_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to cancel", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Registration - Capturing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if sample_count > 0:
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        employee_data = {
            "employee_id": employee_id,
            "name": name,
            "embeddings": embeddings,
            "avg_embedding": avg_embedding,
            "registration_date": datetime.now()
        }
        employees.insert_one(employee_data)
        print(f"Registered {name} successfully with {sample_count} samples!")
    else:
        print("Registration failed - no face samples captured") 