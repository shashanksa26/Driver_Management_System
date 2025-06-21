import cv2
import numpy as np
import pymongo
from pymongo import MongoClient
import mediapipe as mp
import tensorflow as tf
from datetime import datetime
import os

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
employees = db['employees']

# Create database if not exists
if 'face_recognition_db' not in client.list_database_names():
    print("Creating database...")
    db.command("create", "face_recognition_db")
    employees.create_index("employee_id", unique=True)

# Load MobileFaceNet TFLite model
model_path = 'mobilefacenet.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_face(face_img):
    """Preprocess face image for embedding extraction"""
    face_img = cv2.resize(face_img, (160, 160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img.astype(np.float32) - 127.5) * 0.0078125
    return np.expand_dims(face_img, axis=0)

def extract_embedding(face_img):
    """Extract face embedding using MobileFaceNet"""
    input_data = preprocess_face(face_img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def register_new_employee():
    """Register a new employee by capturing multiple face samples"""
    employee_id = input("Enter employee ID: ")
    name = input("Enter employee name: ")
    
    # Check if employee already exists
    if employees.find_one({"employee_id": employee_id}):
        print(f"Employee {employee_id} already exists!")
        return
    
    cap = cv2.VideoCapture(0)
    embeddings = []
    sample_count = 0
    required_samples = 5
    
    print(f"Capturing {required_samples} face samples...")
    
    while sample_count < required_samples:
        ret, frame = cap.read()
        if not ret:
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Get face bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Expand bounding box slightly
                x = max(0, x - 20)
                y = max(0, y - 30)
                w = min(iw - x, w + 40)
                h = min(ih - y, h + 60)
                
                # Extract and process face
                face_img = frame[y:y+h, x:x+w]
                
                # Calculate embedding
                embedding = extract_embedding(face_img)
                embeddings.append(embedding.tolist())
                sample_count += 1
                
                # Display
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Sample: {sample_count}/{required_samples}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Registration - Press 'q' to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if sample_count > 0:
        # Calculate average embedding
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        
        # Store in MongoDB
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

def recognize_faces():
    """Real-time face recognition from camera"""
    # Get all employee embeddings from DB
    employee_data = list(employees.find({}))
    known_embeddings = {emp['employee_id']: np.array(emp['avg_embedding']) for emp in employee_data}
    
    if not known_embeddings:
        print("No employees registered in database!")
        return
    
    cap = cv2.VideoCapture(0)
    recognition_active = True
    recognition_threshold = 0.65
    
    while recognition_active:
        ret, frame = cap.read()
        if not ret:
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Get face bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Expand bounding box slightly
                x = max(0, x - 20)
                y = max(0, y - 30)
                w = min(iw - x, w + 40)
                h = min(ih - y, h + 60)
                
                # Extract face
                face_img = frame[y:y+h, x:x+w]
                
                # Calculate embedding
                embedding = extract_embedding(face_img)
                
                # Find best match
                best_match = None
                best_similarity = 0
                
                for emp_id, ref_embedding in known_embeddings.items():
                    similarity = cosine_similarity(embedding, ref_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = emp_id
                
                # Display results
                color = (0, 255, 0) if best_similarity > recognition_threshold else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                if best_similarity > recognition_threshold:
                    emp = employees.find_one({"employee_id": best_match})
                    name = emp['name']
                    cv2.putText(frame, f"{name} ({best_match})", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Confidence: {best_similarity:.2f}", (x, y+h+30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Face Recognition - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recognition_active = False
    
    cap.release()
    cv2.destroyAllWindows()

# Main menu
def main_menu():
    while True:
        print("\nDriver Monitoring System")
        print("1. Register New Employee")
        print("2. Real-time Face Recognition")
        print("3. Exit")
        
        choice = input("Select option: ")
        
        if choice == '1':
            register_new_employee()
        elif choice == '2':
            recognize_faces()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main_menu()