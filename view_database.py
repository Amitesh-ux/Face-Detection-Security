"""
Database Viewer - Shows what's inside face_database.pkl in readable format
"""
import pickle
import os

DATABASE_FILE = 'face_database.pkl'

print("="*60)
print("FACE DATABASE VIEWER")
print("="*60)

if not os.path.exists(DATABASE_FILE):
    print("\n❌ No database file found!")
    print("   Make sure 'face_database.pkl' is in the same folder as this script.")
    exit()

# Load the database
with open(DATABASE_FILE, 'rb') as f:
    known_faces = pickle.load(f)

print(f"\n📊 Database Summary:")
print(f"   Total people enrolled: {len(known_faces)}")
print()

# Show details for each person
print("="*60)
print("ENROLLED PEOPLE:")
print("="*60)

for name, encodings in known_faces.items():
    print(f"\n👤 Name: {name}")
    print(f"   Number of photos: {len(encodings)}")
    print(f"   Data structure: List of {len(encodings)} encoding(s)")
    
    # Show details of each photo's encoding
    for i, encoding in enumerate(encodings):
        print(f"\n   📸 Photo {i+1}:")
        print(f"      - Type: {type(encoding).__name__}")
        print(f"      - Length: {len(encoding)} numbers")
        print(f"      - First 5 numbers: {encoding[:5]}")
        print(f"      - Last 5 numbers: {encoding[-5:]}")
        
        # Verify it's valid
        if isinstance(encoding, list) and len(encoding) == 128:
            print(f"      - Status: ✅ Valid encoding")
        else:
            print(f"      - Status: ❌ INVALID!")

print("\n" + "="*60)
print("END OF DATABASE")
print("="*60)