# Face Recognition Security System

A real-time face recognition system with webcam integration, multiple visual filters, and a simple enrollment interface. Built with Python, OpenCV, and face_recognition library.

## Features

- **Face Enrollment System**: Easy-to-use interface to register authorized faces
- **Real-time Recognition**: Identifies enrolled faces through webcam feed
- **8 Visual Filters**: Apply different effects (grayscale, sepia, blur, edge detection, negative, bright, cartoon)
- **Snapshot Capture**: Save filtered images with timestamps
- **Persistent Database**: Face encodings stored locally using pickle

## Demo

[Add screenshots or GIF demo here if you'd like]

## Prerequisites

- Python 3.7+
- Webcam
- CMake (required for dlib installation)

### For Windows:
- Visual Studio Build Tools or Visual Studio Community
- CMake

### For macOS:
```bash
brew install cmake
```

### For Linux:
```bash
sudo apt-get install cmake
sudo apt-get install python3-dev
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-security.git
cd face-recognition-security
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install opencv-python
pip install face_recognition
pip install numpy
```

**Note**: Installing `face_recognition` also installs `dlib`, which requires CMake and can take several minutes.

## Usage

### 1. Enroll Faces

First, register authorized faces in the system:

```bash
python enroll_faces.py
```

**Menu Options:**
- `1` - Enroll new person (press SPACE to capture, ESC to cancel)
- `2` - List all enrolled people
- `3` - Delete a person from database
- `4` - Exit

### 2. Run Face Detection

Start the real-time face recognition system:

```bash
python face_detect.py
```

**Controls:**
- `0-7` - Switch between visual filters
- `s` - Save snapshot
- `q` or `ESC` - Quit

## How It Works

### Face Enrollment (`enroll_faces.py`)
1. Captures webcam image when SPACE is pressed
2. Detects face and extracts 128-dimensional face encoding
3. Stores encoding with person's name in `face_database.pkl`
4. Validates that exactly one face is visible

### Face Recognition (`face_detect.py`)
1. Loads enrolled faces from database
2. Processes webcam feed every 15 frames for performance
3. Compares detected faces against known encodings
4. Draws green box for recognized faces, red for unknown
5. Applies selected visual filter to the display

### Visual Filters
- **Normal** - Original feed
- **Grayscale** - Black and white
- **Sepia** - Vintage brown tone
- **Blur** - Gaussian blur effect
- **Edge Detection** - Canny edge detection
- **Negative** - Inverted colors
- **Bright** - Enhanced brightness
- **Cartoon** - Cartoon-style effect

## Project Structure

```
.
├── enroll_faces.py           # Face enrollment interface
├── face_detect.py            # Real-time recognition with filters
├── test_opencv.py            # OpenCV installation test
├── test_face_recognition.py  # face_recognition installation test
├── face_database.pkl         # Stored face encodings (auto-generated)
├── .gitignore               # Excludes sensitive data
└── README.md                # This file
```

## Security & Privacy

⚠️ **Important Notes:**

- Face encodings are stored locally in `face_database.pkl`
- This file contains biometric data and should NEVER be committed to version control
- The `.gitignore` file excludes the database and captured images
- This is a demonstration project and not suitable for production security systems
- Always comply with local privacy laws when using facial recognition

## Troubleshooting

### dlib installation fails
If `pip install face_recognition` fails:
- Ensure CMake is installed
- On Windows, install Visual Studio Build Tools
- Try installing dlib separately: `pip install dlib`

### Webcam not detected
- Check that your webcam is connected and not in use by another application
- Try changing the camera index in code: `cv2.VideoCapture(1)` instead of `0`

### Recognition not working
- Ensure faces were enrolled successfully
- Check lighting conditions (good lighting improves accuracy)
- Adjust tolerance in `face_detect.py` (line with `tolerance=0.5`)

### Poor performance
- The system processes every 15th frame by default
- Increase this number in `face_detect.py` (line: `if frame_count % 15 == 0`)
- Reduce webcam resolution

## Technical Details

- **Face Detection**: Uses Histogram of Oriented Gradients (HOG) or CNN
- **Face Encoding**: Generates 128-dimensional face embeddings using dlib's ResNet model
- **Recognition Method**: Compares face encodings using Euclidean distance
- **Default Tolerance**: 0.5 (lower = stricter matching)

## License

[Add your chosen license here - MIT, Apache 2.0, etc.]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- Uses [OpenCV](https://opencv.org/) for image processing
- Powered by [dlib](http://dlib.net/) machine learning library

## Disclaimer

This project is for educational purposes only. Facial recognition technology should be used responsibly and in compliance with applicable laws and regulations. The authors are not responsible for any misuse of this software.