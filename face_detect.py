import cv2
import numpy as np
from datetime import datetime
import face_recognition
import pickle
import os

# Load the face database
DATABASE_FILE = 'face_database.pkl'
known_faces = {}
known_names = []
known_encodings = []

if os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'rb') as f:
        known_faces = pickle.load(f)
    
    # NEW: Handle multiple encodings per person
    # Old code: known_encodings = [encoding1, encoding2, ...]  (one per person)
    # New code: Flatten all encodings and track which person each belongs to
    
    for name, encodings_list in known_faces.items():
        # Each person now has a LIST of encodings (3 photos)
        for encoding in encodings_list:
            known_encodings.append(encoding)  # Add this encoding to the flat list
            known_names.append(name)  # Remember this encoding belongs to this person
    
    # Example result:
    # known_names =     ["John", "John", "John", "Jane", "Jane", "Jane"]
    # known_encodings = [enc1,   enc2,   enc3,   enc4,   enc5,   enc6  ]
    
    print(f"✓ Loaded {len(known_faces)} authorized face(s)")
    print(f"  Total reference photos: {len(known_encodings)}")
else:
    print("⚠ No face database found. Run enroll_faces.py first!")

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Filter functions
def apply_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_sepia(frame):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

def apply_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_negative(frame):
    return cv2.bitwise_not(frame)

def apply_bright(frame):
    return cv2.convertScaleAbs(frame, alpha=1.3, beta=30)

def apply_cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges)
    return cartoon

# Filter dictionary
filters = {
    0: ("Normal", lambda x: x),
    1: ("Grayscale", apply_grayscale),
    2: ("Sepia", apply_sepia),
    3: ("Blur", apply_blur),
    4: ("Edge Detection", apply_edge_detection),
    5: ("Negative", apply_negative),
    6: ("Bright", apply_bright),
    7: ("Cartoon", apply_cartoon)
}

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened!")
print("Press number keys 0-7 to change filters")
print("Press 's' to save snapshot")
print("Press 'q' to quit")

snapshot_count = 0
current_filter = 0
frame_count = 0
recognized_faces = []  # FIXED: Moved outside the loop

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Apply current filter
    filter_name, filter_func = filters[current_filter]
    filtered_frame = filter_func(frame.copy())
    
    # Run face recognition every 30 frames (reduced from 15 for better performance)
    # Higher number = less lag but slower detection updates
    frame_count += 1
    if frame_count % 30 == 0 and len(known_encodings) > 0:
        # Use smaller frame for faster processing (0.25 = 1/4 size for even faster processing)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using HOG model (faster than CNN)
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        
        # Clear old results
        recognized_faces = []
        
        # Check each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale coordinates back (we used 0.25x frame, so multiply by 4)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Compare with known faces
            # NOTE: known_encodings now contains ALL photos for ALL people (flattened)
            # If person has 3 photos, we check against all 3 automatically
            # Example: ["John", "John", "John", "Jane", "Jane", "Jane"]
            #          If matches[1] is True, it matched John's 2nd photo
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "UNKNOWN"
            color = (0, 0, 255)  # Red
            
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]  # Get the person's name for this encoding
                color = (0, 255, 0)  # Green
            
            # Store result
            recognized_faces.append((name, color, left, top, right, bottom))
    
    # FIXED: Draw faces (outside the recognition block)
    for name, color, left, top, right, bottom in recognized_faces:
        cv2.rectangle(filtered_frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(filtered_frame, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
        cv2.putText(filtered_frame, name, (left + 6, bottom + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # FIXED: Display info (outside the loop)
    cv2.putText(filtered_frame, f'Filter: {filter_name}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(filtered_frame, f'Authorized: {len(known_faces)}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(filtered_frame, "0-7: Filters | 's': Save | 'q' or 'Esc': Quit", 
                (10, filtered_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # FIXED: Display the frame (outside the loop)
    cv2.imshow('Security Face Recognition', filtered_frame)
    
    # FIXED: Check for key presses (outside the loop)
    key = cv2.waitKey(1) & 0xFF
    
    if key >= ord('0') and key <= ord('7'):
        current_filter = key - ord('0')
        print(f"Filter changed to: {filters[current_filter][0]}")
    elif key == ord('s'):
        snapshot_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'snapshot_{snapshot_count}_{filter_name}_{timestamp}.jpg'
        cv2.imwrite(filename, filtered_frame)
        print(f"✓ Saved {filename}")
    elif key == ord('q') or key == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print(f"\nTotal snapshots saved: {snapshot_count}")
print("Webcam closed")