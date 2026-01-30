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
    
    # BACKWARD COMPATIBILITY: Convert old format to new format
    # Old format: known_faces["John"] = [0.1, 0.2, ..., 0.5]  (single encoding)
    # New format: known_faces["John"] = [[0.1, 0.2, ...], [0.3, 0.4, ...]]  (list of encodings)
    for name in list(known_faces.keys()):
        # Check if this person's data is in old format (single encoding)
        if len(known_faces[name]) > 0 and isinstance(known_faces[name][0], (float, int)):
            # Convert: wrap the single encoding in a list to make it a list of encodings
            known_faces[name] = [known_faces[name]]
            print(f"  Converted {name} to new format (1 photo)")
    
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
    print("We'll capture 3 photos for better accuracy")
    print("Press SPACE to capture each photo, ESC to cancel")
    
    # List to store all 3 encodings for this person
    encodings_list = []
    photos_captured = 0
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while photos_captured < 3:  # Loop until we have 3 photos
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break
        
        # Display instructions on screen - show which photo number we're on
        instruction_text = f"Photo {photos_captured + 1}/3 - Press SPACE to capture"
        cv2.putText(frame, instruction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show tips for different photos
        if photos_captured == 0:
            cv2.putText(frame, "Look straight at camera", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        elif photos_captured == 1:
            cv2.putText(frame, "Turn head slightly left/right", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Different expression (smile/neutral)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
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
                # Save this encoding to our list
                encodings_list.append(face_encodings[0])
                photos_captured += 1
                print(f"✓ Photo {photos_captured}/3 captured!")
                
                # If we have all 3 photos, we're done!
                if photos_captured == 3:
                    # Store the list of 3 encodings for this person
                    known_faces[name] = encodings_list
                    print(f"✓ {name} enrolled successfully with 3 photos!")
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
        # Show all enrolled names with photo count
        if known_faces:
            print("\nEnrolled people:")
            for name in known_faces.keys():
                num_photos = len(known_faces[name])  # Count how many photos this person has
                print(f"  - {name} ({num_photos} photo{'s' if num_photos != 1 else ''})")
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