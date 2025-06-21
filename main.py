import mediapipe as mp
from register import register_new_employee
from recognize import recognize_faces

def main_menu(face_detection):
    while True:
        print("\nDriver Monitoring System")
        print("1. Register New Employee")
        print("2. Real-time Face Recognition")
        print("3. Exit")
        choice = input("Select option: ")
        if choice == '1':
            register_new_employee(face_detection)
        elif choice == '2':
            recognize_faces(face_detection)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    main_menu(face_detection) 