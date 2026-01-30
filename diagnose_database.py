"""
Detailed Database Diagnostic - Check what's actually stored
"""
import pickle
import os
import numpy as np

DATABASE_FILE = 'face_database.pkl'

print("="*60)
print("DETAILED DATABASE DIAGNOSTIC")
print("="*60)

if not os.path.exists(DATABASE_FILE):
    print("\n❌ No database file found!")
    exit()

# Load the database
with open(DATABASE_FILE, 'rb') as f:
    known_faces = pickle.load(f)

print(f"\n📊 Total people: {len(known_faces)}")

for name, encodings in known_faces.items():
    print(f"\n{'='*60}")
    print(f"👤 Name: {name}")
    print(f"   Photos: {len(encodings)}")
    print(f"   Encodings type: {type(encodings)}")
    print(f"   Encodings is list?: {isinstance(encodings, list)}")
    
    # Check each encoding
    for i, encoding in enumerate(encodings):
        print(f"\n   📸 Photo {i+1} DETAILS:")
        print(f"      Type: {type(encoding)}")
        print(f"      Is list?: {isinstance(encoding, list)}")
        print(f"      Is numpy array?: {isinstance(encoding, np.ndarray)}")
        
        if hasattr(encoding, '__len__'):
            print(f"      Length: {len(encoding)}")
            print(f"      Shape (if array): {encoding.shape if hasattr(encoding, 'shape') else 'N/A'}")
            print(f"      First element type: {type(encoding[0])}")
            print(f"      First 3 values: {encoding[:3]}")
        else:
            print(f"      ❌ No length attribute!")
        
        # Try to determine if it's valid
        is_valid = False
        if isinstance(encoding, list) and len(encoding) == 128:
            is_valid = True
            print(f"      ✅ Valid Python list (128 numbers)")
        elif isinstance(encoding, np.ndarray) and encoding.shape == (128,):
            is_valid = True
            print(f"      ✅ Valid numpy array (128 numbers)")
        else:
            print(f"      ❌ INVALID FORMAT")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)