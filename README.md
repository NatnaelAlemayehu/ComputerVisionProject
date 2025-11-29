# Computer Vision Course - Unified Web Application

A unified Flask web application that combines all computer vision course assignments into a single, easy-to-use interface.

## 🎯 Overview

This application integrates five different computer vision assignments:

1. **Assignment 1: Camera Calibration** - Real-world dimension estimation from images
2. **Assignment 2: Template Matching** - Object detection and FFT-based blurring
3. **Assignment 3: Image Processing** - Gradients, edges, corners, and contours
4. **Assignment 4: SIFT & Image Stitching** - Feature extraction and panorama creation
5. **Assignment 7: Real-time Tracking** - Live pose and hand landmark detection

## 📋 Requirements

- Python 3.8 or higher
- Webcam (for Assignment 7 only)

## 🚀 Installation

1. **Navigate to the Combined folder:**
   ```bash
   cd "/Users/nati/Documents/Computer Vision Course/assignments/Combined"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **You should see the home page with links to all five assignments.**

## 📚 Assignment Details

### Assignment 1: Camera Calibration
**URL:** `http://localhost:5000/assignment1`

**Features:**
- Upload an image
- Click two points to measure distances
- Uses camera calibration parameters to convert pixel distances to real-world measurements
- Adjust focal length and camera-to-plane distance

**Usage:**
1. Upload an image
2. Set focal length (fx) and distance (D) parameters
3. Click two points on the image to measure the distance between them

### Assignment 2: Template Matching
**URL:** `http://localhost:5000/assignment2`

**Features:**
- Upload a scene image
- Automatically detects objects using pre-loaded templates
- Supports multiple matching algorithms (TM_CCOEFF, TM_CCOEFF_NORMED, etc.)
- Applies FFT-based Gaussian blur to detected regions

**Usage:**
1. Select a matching method
2. Upload an image
3. View detection results and blurred output

**Note:** Place template images in `static/template/` folder for detection.

### Assignment 3: Image Processing
**URL:** `http://localhost:5000/assignment3`

**Features:**
- **Question 1:** Gradient magnitude, angle, and Laplacian of Gaussian (LoG)
- **Question 2a:** Edge detection with adjustable threshold
- **Question 2b:** Harris corner detection with parameter tuning
- **Question 3:** Contour detection using Otsu's or Canny methods

**Usage:**
1. Navigate between tabs for different processing modes
2. Upload an image
3. Adjust parameters as needed
4. Process and view results

### Assignment 4: SIFT & Image Stitching
**URL:** `http://localhost:5000/assignment4`

**Features:**
- **Image Stitching:** Create panoramas from 2+ images
- **SIFT Comparison:** Compare custom SIFT implementation with OpenCV
- View keypoints, matches, and inliers
- Adjustable matching parameters

**Usage:**

**For Stitching:**
1. Select multiple images (minimum 2)
2. Click "Stitch Images"
3. View the panorama result

**For SIFT:**
1. Upload two images
2. Adjust ratio threshold, RANSAC threshold, and max matches
3. Click "Compare SIFT"
4. View side-by-side comparison of custom vs OpenCV SIFT

### Assignment 7: Real-time Tracking
**URL:** `http://localhost:5000/assignment7`

**Features:**
- Real-time pose estimation (33 body landmarks)
- Hand tracking (21 landmarks per hand, up to 2 hands)
- Record tracking data
- Export landmarks as CSV

**Usage:**
1. Allow camera access when prompted
2. Click "Start Recording" to begin capturing landmarks
3. Click "Stop Recording" when done
4. Click "Download CSV" to export the tracking data

**CSV Format:**
- Frame number and timestamp
- Pose landmarks (x, y, z, visibility)
- Left hand landmarks (x, y, z)
- Right hand landmarks (x, y, z)

## 📁 Project Structure

```
Combined/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── sift.py                     # Custom SIFT implementation
├── README.md                   # This file
│
├── templates/                  # HTML templates
│   ├── index.html             # Home page
│   ├── assignment1.html       # Assignment 1 page
│   ├── assignment2.html       # Assignment 2 page
│   ├── assignment3.html       # Assignment 3 page
│   ├── assignment4.html       # Assignment 4 page
│   └── assignment7.html       # Assignment 7 page
│
├── static/                     # Static files
│   ├── css/
│   │   └── assignment3.css    # Assignment 3 styles
│   ├── js/
│   │   └── assignment3.js     # Assignment 3 scripts
│   ├── calibration.json       # Camera calibration data
│   ├── template/              # Template images for matching
│   └── output/                # Processed images output
│       ├── gradients/
│       ├── edges/
│       ├── corners/
│       ├── contours/
│       ├── sift/
│       └── stitching/
│
└── uploads/                    # Temporary uploaded files
```

## 🔧 Configuration

### Camera Calibration (Assignment 1)

Edit `static/calibration.json` to set your camera parameters:
```json
{
  "fx": 1500.0,
  "fy": 1500.0,
  "cx": 640.0,
  "cy": 360.0
}
```

### Template Images (Assignment 2)

Place your template images in `static/template/` folder. Supported formats: JPG, PNG.

## 🛠️ Troubleshooting

### MediaPipe Not Working (Assignment 7)
If Assignment 7 shows an error:
```bash
pip install --upgrade mediapipe
```

### Camera Access Issues
- Ensure your browser has camera permissions enabled
- Check if another application is using the camera
- Try a different browser (Chrome/Firefox recommended)

### Template Matching Not Finding Templates
- Verify template images are in `static/template/` folder
- Ensure templates are in JPG or PNG format
- Templates should not be nearly blank (minimum variance threshold)

### SIFT Module Not Found
The custom SIFT implementation (`sift.py`) should be in the Combined folder. If missing, the app will fall back to OpenCV SIFT only.

## 🎨 Features

- **Unified Interface:** All assignments accessible from a single home page
- **Modern UI:** Clean, responsive design with gradient themes
- **Real-time Processing:** Instant feedback for all operations
- **Easy Navigation:** Back to home button on every page
- **Error Handling:** Clear error messages for debugging

## 📊 Supported Image Formats

- JPG/JPEG
- PNG
- GIF
- BMP

## 🔒 Security Notes

- Maximum upload size: 50MB per file
- Only image files are accepted
- Temporary files are stored in `uploads/` folder
- Consider clearing `uploads/` and `static/output/` periodically

## 💡 Tips

1. **Image Size:** Larger images take longer to process. Consider resizing very large images.
2. **Template Matching:** Works best with distinctive templates and good lighting.
3. **Image Stitching:** Images should have overlapping regions (30-50% overlap recommended).
4. **SIFT Matching:** Adjust ratio threshold (0.6-0.8) for better matches.
5. **Real-time Tracking:** Ensure good lighting for optimal landmark detection.

## 📝 Development Notes

- The app runs in debug mode by default (don't use in production)
- Each assignment maintains its original functionality
- Routes are prefixed with `/assignmentN/` for organization
- Static files use `/static/` prefix
- Uploads use `/uploads/` prefix

## 🤝 Credits

Computer Vision Course - All Assignments
Built with Flask, OpenCV, MediaPipe, and NumPy

## 📄 License

Educational use only - Computer Vision Course

---

**Enjoy exploring computer vision concepts!** 🎓
