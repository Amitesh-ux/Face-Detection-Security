import face_recognition
import cv2
import pickle  # For saving/loading data to files
import os

# File to store face encodings (our "database")
DATABASE_FILE = 'face_database.pkl'

# Load existing database or create new one
# This checks if we already have enrolled faces saved
if os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'rb') as f:
        known_faces = pickle.load(f)  # Load the dictionary of names -> encodings
    print(f"Loaded {len(known_faces)} existing face(s)")
else:
    known_faces = {}  # Empty dictionary to start fresh
    print("Creating new face database")

def enroll_person():
    """Function to enroll a new person into the system"""
    
    # Get the person's name
    name = input("Enter person's name: ").strip()
    
    # Check if this person already exists
    if name in known_faces:
        overwrite = input(f"{name} already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            return
    
    print(f"\nEnrolling {name}...")
    print("Press SPACE to capture photo, ESC to cancel")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break
        
        # Display instructions on screen
        cv2.putText(frame, "Press SPACE to capture", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Enrollment', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # SPACE key pressed
            # Convert BGR (OpenCV format) to RGB (face_recognition format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract face encodings from the image
            # face_encodings returns a list of 128-dimensional vectors (one per face detected)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            
            # Validation: make sure exactly one face is detected
            if len(face_encodings) == 0:
                print("No face detected! Try again.")
                continue
            elif len(face_encodings) > 1:
                print("Multiple faces detected! Make sure only one person is visible.")
                continue
            else:
                # Save the face encoding (128-number representation of the face)
                known_faces[name] = face_encodings[0]
                print(f"✓ {name} enrolled successfully!")
                break
        
        elif key == 27:  # ESC key pressed - cancel enrollment
            print("Enrollment cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    # Clean up webcam
    cap.release()
    cv2.destroyAllWindows()
    
    # Save the updated database to disk
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(known_faces, f)  # Serialize the dictionary
    print(f"Database saved with {len(known_faces)} face(s)")

# Main menu loop
while True:
    print("\n=== Face Enrollment System ===")
    print("1. Enroll new person")
    print("2. List enrolled people")
    print("3. Delete person")
    print("4. Exit")
    
    choice = input("Choose option: ").strip()
    
    if choice == '1':
        enroll_person()
    elif choice == '2':
        # Show all enrolled names
        if known_faces:
            print("\nEnrolled people:")
            for name in known_faces.keys():
                print(f"  - {name}")
        else:
            print("No people enrolled yet")
    elif choice == '3':
        # Delete a person from database
        name = input("Enter name to delete: ").strip()
        if name in known_faces:
            del known_faces[name]
            # Save updated database
            with open(DATABASE_FILE, 'wb') as f:
                pickle.dump(known_faces, f)
            print(f"✓ {name} deleted")
        else:
            print(f"{name} not found")
    elif choice == '4':
        break  # Exit the program