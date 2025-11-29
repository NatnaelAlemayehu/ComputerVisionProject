"""
Unified Computer Vision Course Web Application
Combines all assignment web applications into a single Flask app
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response, send_file
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import sys
import json
import csv
import time
import base64
import threading
from io import BytesIO, StringIO
from werkzeug.utils import secure_filename
from pathlib import Path
from numpy.fft import fftn, ifftn, fftfreq
import multiprocessing

# On macOS and some Python installs the default multiprocessing start method is
# 'spawn' which can lead to resource_tracker warnings about leaked semaphore
# objects at shutdown when libraries create inter-process semaphores.
# Setting the start method to 'fork' early can avoid these warnings in many
# cases. Guard with try/except because the start method can only be set once.
try:
    multiprocessing.set_start_method('fork')
except RuntimeError:
    # start method already set elsewhere; ignore
    pass

# Import MediaPipe for Assignment 7
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Assignment 7 features will be disabled.")

# Import SIFT module for Assignment 4
try:
    from sift import SIFT, RANSAC, match_features
    SIFT_AVAILABLE = True
except ImportError:
    SIFT_AVAILABLE = False
    print("Warning: SIFT module not found. Assignment 4 SIFT features will use OpenCV only.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configure folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['STATIC_FOLDER'] = os.path.join(BASE_DIR, 'static')

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'output'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'output', 'gradients'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'output', 'edges'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'output', 'corners'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'output', 'contours'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'output', 'sift'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'output', 'stitching'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'template'), exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables for Assignment 7 (Real-time tracking)
camera = None
pose = None
hands = None
mp_drawing = None
mp_styles = None
is_recording = False
csv_data = []
frame_idx = 0
output_frame = None
lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# HOME PAGE
# ============================================================================

@app.route('/')
def index():
    """Main landing page with navigation to all assignments"""
    return render_template('index.html')


# ============================================================================
# ASSIGNMENT 1: Camera Calibration & Real-World Dimension Estimation
# ============================================================================

@app.route('/assignment1')
def assignment1():
    """Camera calibration and dimension estimation page"""
    return render_template('assignment1.html')

@app.route('/assignment1/calibration')
def get_calibration():
    """Serve calibration data"""
    try:
        calibration_path = os.path.join(app.config['STATIC_FOLDER'], 'calibration.json')
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)
        return jsonify(calibration_data)
    except FileNotFoundError:
        return jsonify({'error': 'Calibration file not found'}), 404

@app.route('/assignment1/upload', methods=['POST'])
def upload_image():
    """Handle image upload for Assignment 1"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'filepath': f'/uploads/{filename}'})
    
    return jsonify({'error': 'Upload failed'}), 500


# ============================================================================
# ASSIGNMENT 2: Template Matching & Blurring
# ============================================================================

# Template matching methods
MATCHING_METHODS = {
    'TM_CCOEFF': cv2.TM_CCOEFF,
    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
    'TM_CCORR': cv2.TM_CCORR,
    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    'TM_SQDIFF': cv2.TM_SQDIFF,
    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
}

def load_templates(template_dir=None, min_var=5.0):
    """Load all templates as RGB images, ignoring nearly blank ones."""
    if template_dir is None:
        template_dir = os.path.join(app.config['STATIC_FOLDER'], 'template')
    
    templates = []
    if not os.path.exists(template_dir):
        return templates
    
    for file in sorted(os.listdir(template_dir)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(template_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if np.var(rgb.astype(np.float32)) < min_var:
            continue
        templates.append((file, rgb))
    return templates

def calc_kxky(img):
    """Compute spatial frequency coordinates."""
    h, w = img.shape[:2]
    kx = 2.0 * np.pi * fftfreq(h)
    ky = 2.0 * np.pi * fftfreq(w)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    return kx, ky

def fft_blur_region(img_rgb, x, y, w, h, gaussian_strength=100.0):
    """Apply FFT-based Gaussian blur to a region."""
    H, W = img_rgb.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    roi = img_rgb[y:y+h, x:x+w].astype(np.float64) / 255.0
    kx, ky = calc_kxky(roi)
    psf_fft = np.exp(-gaussian_strength * (kx**2 + ky**2))

    roi_fft = fftn(roi, axes=(0, 1))
    for c in range(roi.shape[2]):
        roi_fft[:, :, c] = roi_fft[:, :, c] * psf_fft

    roi_blurred = np.real(ifftn(roi_fft, axes=(0, 1)))
    roi_blurred -= roi_blurred.min()
    roi_blurred /= roi_blurred.max()
    roi_blurred = (roi_blurred * 255).astype(np.uint8)

    img_rgb[y:y+h, x:x+w] = roi_blurred
    return img_rgb

def best_single_template_match(scene_rgb, templates, method=cv2.TM_CCOEFF_NORMED):
    """Run template matching with the specified method for all templates."""
    vis = scene_rgb.copy()
    best_score = -np.inf if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else np.inf
    best_box = None
    best_name = None
    debug_scores = []

    use_min = method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    for name, templ in templates:
        h, w, _ = templ.shape

        if h > scene_rgb.shape[0] or w > scene_rgb.shape[1]:
            debug_scores.append((name, None))
            continue

        res = cv2.matchTemplate(scene_rgb, templ, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if use_min:
            score = float(min_val)
            loc = min_loc
            debug_scores.append((name, score))
            
            if score < best_score:
                best_score = score
                best_name = name
                best_box = (loc[0], loc[1], w, h)
        else:
            score = float(max_val)
            loc = max_loc
            debug_scores.append((name, score))
            
            if score > best_score:
                best_score = score
                best_name = name
                best_box = (loc[0], loc[1], w, h)

    if best_box:
        x, y, w, h = best_box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(vis, f"Best: {best_name}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return best_box, best_name, vis, debug_scores, best_score

def blur_best_region(scene_rgb, best_box):
    """Blur only the detected best region."""
    if not best_box:
        return scene_rgb
    x, y, w, h = best_box
    return fft_blur_region(scene_rgb.copy(), x, y, w, h)

def image_to_base64(img_rgb):
    """Convert RGB image to base64 string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

@app.route('/assignment2')
def assignment2():
    """Template matching and blurring page"""
    templates = load_templates()
    template_names = [name for name, _ in templates]
    return render_template('assignment2.html', 
                         matching_methods=list(MATCHING_METHODS.keys()),
                         templates=template_names)

@app.route('/assignment2/process', methods=['POST'])
def process_template_matching():
    """Process the uploaded image with template matching and blurring."""
    try:
        method_name = request.form.get('method', 'TM_CCOEFF_NORMED')
        method = MATCHING_METHODS.get(method_name, cv2.TM_CCOEFF_NORMED)
        
        templates = load_templates()
        if not templates:
            return jsonify({'error': 'No valid templates found'}), 400
        
        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        scene_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if scene_bgr is None:
            return jsonify({'error': 'Could not decode image'}), 400
        scene_rgb = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB)
        
        best_box, best_name, vis_img, scores, best_score = best_single_template_match(
            scene_rgb, templates, method
        )
        blurred_img = blur_best_region(scene_rgb, best_box)
        
        original_b64 = image_to_base64(scene_rgb)
        detection_b64 = image_to_base64(vis_img)
        blurred_b64 = image_to_base64(blurred_img)
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'detection': detection_b64,
            'blurred': blurred_b64,
            'best_match': best_name,
            'best_score': best_score,
            'method': method_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ASSIGNMENT 3: Image Processing (Gradients, Edges, Corners, Contours)
# ============================================================================

@app.route('/assignment3')
def assignment3():
    """Image processing page"""
    return render_template('assignment3.html')

@app.route('/assignment3/process_gradients', methods=['POST'])
def process_gradients():
    """Process gradients and Laplacian of Gaussian"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    
    magnitude_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle_vis = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log_vis = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    base_name = os.path.splitext(filename)[0]
    magnitude_path = os.path.join('gradients', f'{base_name}_magnitude.jpg')
    angle_path = os.path.join('gradients', f'{base_name}_angle.jpg')
    log_path = os.path.join('gradients', f'{base_name}_log.jpg')
    
    cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], 'output', magnitude_path), magnitude_vis)
    cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], 'output', angle_path), angle_vis)
    cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], 'output', log_path), log_vis)
    
    return jsonify({
        'magnitude': f'/static/output/{magnitude_path}',
        'angle': f'/static/output/{angle_path}',
        'log': f'/static/output/{log_path}',
        'original': f'/uploads/{filename}'
    })

@app.route('/assignment3/process_edges', methods=['POST'])
def process_edges():
    """Process edge detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    threshold = int(request.form.get('threshold', 80))
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, edges = cv2.threshold(magnitude_vis, threshold, 255, cv2.THRESH_BINARY)
    
    base_name = os.path.splitext(filename)[0]
    edges_path = os.path.join('edges', f'{base_name}_edges.jpg')
    
    cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], 'output', edges_path), edges)
    
    return jsonify({
        'edges': f'/static/output/{edges_path}',
        'original': f'/uploads/{filename}'
    })

@app.route('/assignment3/process_corners', methods=['POST'])
def process_corners():
    """Process corner detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    max_corners = int(request.form.get('maxCorners', 200))
    quality_level = float(request.form.get('qualityLevel', 0.01))
    min_distance = int(request.form.get('minDistance', 20))
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=3,
        useHarrisDetector=True,
        k=0.04
    )
    
    output_img = img.copy()
    if corners is not None:
        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            cv2.circle(output_img, (x, y), 5, (0, 0, 255), -1)
    
    base_name = os.path.splitext(filename)[0]
    corners_path = os.path.join('corners', f'{base_name}_corners.jpg')
    
    cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], 'output', corners_path), output_img)
    
    corner_count = len(corners) if corners is not None else 0
    
    return jsonify({
        'corners': f'/static/output/{corners_path}',
        'original': f'/uploads/{filename}',
        'count': corner_count
    })

@app.route('/assignment3/process_contours', methods=['POST'])
def process_contours():
    """Process contour detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    method = request.form.get('method', 'otsu')
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == 'canny':
        binary = cv2.Canny(gray_blur, 100, 200)
    else:
        _, binary = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = img.copy()
    cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)
    
    base_name = os.path.splitext(filename)[0]
    binary_path = os.path.join('contours', f'{base_name}_binary.jpg')
    contours_path = os.path.join('contours', f'{base_name}_contours.jpg')
    
    cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], 'output', binary_path), binary)
    cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], 'output', contours_path), output_img)
    
    return jsonify({
        'binary': f'/static/output/{binary_path}',
        'contours': f'/static/output/{contours_path}',
        'original': f'/uploads/{filename}',
        'count': len(contours)
    })

@app.route('/assignment3/process_aruco', methods=['POST'])
def process_aruco():
    """Process ArUco marker detection and segmentation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    buffer_distance = int(request.form.get('buffer', 55))
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        img = cv2.imread(filepath)
        h, w = img.shape[:2]
        
        # ArUco detection
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        
        corners, ids, _ = detector.detectMarkers(img)
        
        if ids is None:
            return jsonify({'error': 'No ArUco markers detected'}), 400
        
        # Get marker centers
        marker_centers = []
        for c in corners:
            center = c[0].mean(axis=0)
            marker_centers.append(center)
        
        marker_centers = np.array(marker_centers, dtype=np.float32)
        
        # Create convex hull
        hull = cv2.convexHull(marker_centers.astype(np.int32))
        
        # Add buffer to expand the boundary outward
        mask_temp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_temp, [hull], 255)
        kernel = np.ones((buffer_distance*2, buffer_distance*2), np.uint8)
        mask_expanded = cv2.dilate(mask_temp, kernel, iterations=1)
        
        # Get the expanded contour
        contours, _ = cv2.findContours(mask_expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expanded_hull = contours[0]
        
        # Draw the boundary on a copy of the image
        boundary_image = img.copy()
        # Draw detected ArUco markers
        aruco.drawDetectedMarkers(boundary_image, corners, ids)
        # Draw the expanded boundary
        cv2.polylines(boundary_image, [expanded_hull], True, (0, 255, 0), 3)
        
        # Save the result
        base_name = os.path.splitext(filename)[0]
        output_path = f'{base_name}_aruco.jpg'
        cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], 'output', output_path), boundary_image)
        
        return jsonify({
            'result': f'/static/output/{output_path}',
            'original': f'/uploads/{filename}',
            'markers_detected': len(ids),
            'marker_ids': ids.flatten().tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/assignment3/upload_comparison', methods=['POST'])
def upload_comparison():
    """Upload comparison images for Question 5"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({'filepath': f'/uploads/{filename}'})


# ============================================================================
# ASSIGNMENT 4: SIFT Feature Extraction & Image Stitching
# ============================================================================

@app.route('/assignment4')
def assignment4():
    """SIFT and image stitching page"""
    return render_template('assignment4.html')

@app.route('/assignment4/stitch_images', methods=['POST'])
def stitch_images():
    """Stitch multiple images together"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    
    if len(files) < 2:
        return jsonify({'error': 'Please upload at least 2 images'}), 400
    
    images = []
    filenames = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filenames.append(filename)
            
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    
    if len(images) < 2:
        return jsonify({'error': 'Failed to load images'}), 400
    
    try:
        stitcher = cv2.Stitcher_create()
        status, result = stitcher.stitch(images)
        
        if status != 0:
            error_messages = {
                1: "Not enough keypoints detected in images",
                2: "Homography estimation failed",
                3: "Camera parameters adjustment failed"
            }
            return jsonify({
                'error': f'Stitching failed: {error_messages.get(status, "Unknown error")}'
            }), 400
        
        stitched = result
        
        output_path = os.path.join('stitching', 'stitched_result.jpg')
        cv2.imwrite(
            os.path.join(app.config['STATIC_FOLDER'], 'output', output_path),
            cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
        )
        
        return jsonify({
            'stitched': f'/static/output/{output_path}',
            'num_images': len(images),
            'dimensions': f"{stitched.shape[1]}x{stitched.shape[0]}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Stitching error: {str(e)}'}), 500

@app.route('/assignment4/process_sift', methods=['POST'])
def process_sift():
    """Process SIFT feature extraction and comparison"""
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Please upload two images'}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    ratio = float(request.form.get('ratio', 0.65))
    ransac_threshold = float(request.form.get('ransac_threshold', 3.0))
    max_matches = int(request.form.get('max_matches', 50))
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(filepath1)
        file2.save(filepath2)
        
        img1_bgr = cv2.imread(filepath1)
        img2_bgr = cv2.imread(filepath2)
        
        # Run custom SIFT if available
        if SIFT_AVAILABLE:
            custom_res = run_custom_sift(
                img1_bgr, img2_bgr,
                ratio=ratio,
                ransac_threshold=ransac_threshold,
                max_matches=max_matches
            )
        else:
            custom_res = None
        
        # Run OpenCV SIFT
        opencv_res = run_opencv_sift(
            img1_bgr, img2_bgr,
            ratio=ratio,
            ransac_threshold=ransac_threshold,
            max_matches=max_matches
        )
        
        # Generate visualizations for OpenCV
        vis_opencv_kp1 = visualize_keypoints(img1_bgr, opencv_res['kps1'], max_keypoints=500)
        vis_opencv_kp2 = visualize_keypoints(img2_bgr, opencv_res['kps2'], max_keypoints=500)
        
        cv2.imwrite(
            os.path.join(app.config['STATIC_FOLDER'], 'output', 'sift', 'opencv_kp1.jpg'),
            vis_opencv_kp1
        )
        cv2.imwrite(
            os.path.join(app.config['STATIC_FOLDER'], 'output', 'sift', 'opencv_kp2.jpg'),
            vis_opencv_kp2
        )
        
        if opencv_res['matches'] > 0:
            vis_opencv_matches = visualize_matches(
                img1_bgr, img2_bgr,
                opencv_res['kps1'], opencv_res['kps2'],
                opencv_res['matches_list'],
                opencv_res['inlier_mask'],
                max_matches=100,
                is_opencv=True
            )
            cv2.imwrite(
                os.path.join(app.config['STATIC_FOLDER'], 'output', 'sift', 'opencv_matches.jpg'),
                vis_opencv_matches
            )
        
        # Prepare comparison data
        comparison = {
            'opencv': {
                'keypoints1': opencv_res['keypoints1'],
                'keypoints2': opencv_res['keypoints2'],
                'matches': opencv_res['matches'],
                'inliers': opencv_res['inliers'],
                'time_detect1': round(opencv_res['time_detect1'], 4),
                'time_detect2': round(opencv_res['time_detect2'], 4),
                'match_ratio': round(opencv_res['matches'] / opencv_res['keypoints1'] * 100, 2) if opencv_res['keypoints1'] > 0 else 0,
                'inlier_ratio': round(opencv_res['inliers'] / opencv_res['matches'] * 100, 2) if opencv_res['matches'] > 0 else 0,
                'keypoints_img1': '/static/output/sift/opencv_kp1.jpg',
                'keypoints_img2': '/static/output/sift/opencv_kp2.jpg',
                'matches_img': '/static/output/sift/opencv_matches.jpg' if opencv_res['matches'] > 0 else None
            }
        }
        
        # Add custom SIFT results if available
        if custom_res and SIFT_AVAILABLE:
            vis_custom_kp1 = visualize_keypoints(img1_bgr, custom_res['kps1'], max_keypoints=500)
            vis_custom_kp2 = visualize_keypoints(img2_bgr, custom_res['kps2'], max_keypoints=500)
            
            cv2.imwrite(
                os.path.join(app.config['STATIC_FOLDER'], 'output', 'sift', 'custom_kp1.jpg'),
                vis_custom_kp1
            )
            cv2.imwrite(
                os.path.join(app.config['STATIC_FOLDER'], 'output', 'sift', 'custom_kp2.jpg'),
                vis_custom_kp2
            )
            
            if custom_res['matches'] > 0:
                vis_custom_matches = visualize_matches(
                    img1_bgr, img2_bgr,
                    custom_res['kps1'], custom_res['kps2'],
                    custom_res['matches_list'],
                    custom_res['inlier_mask'],
                    max_matches=100
                )
                cv2.imwrite(
                    os.path.join(app.config['STATIC_FOLDER'], 'output', 'sift', 'custom_matches.jpg'),
                    vis_custom_matches
                )
            
            comparison['custom'] = {
                'keypoints1': custom_res['keypoints1'],
                'keypoints2': custom_res['keypoints2'],
                'matches': custom_res['matches'],
                'inliers': custom_res['inliers'],
                'time_detect1': round(custom_res['time_detect1'], 4),
                'time_detect2': round(custom_res['time_detect2'], 4),
                'match_ratio': round(custom_res['matches'] / custom_res['keypoints1'] * 100, 2) if custom_res['keypoints1'] > 0 else 0,
                'inlier_ratio': round(custom_res['inliers'] / custom_res['matches'] * 100, 2) if custom_res['matches'] > 0 else 0,
                'keypoints_img1': '/static/output/sift/custom_kp1.jpg',
                'keypoints_img2': '/static/output/sift/custom_kp2.jpg',
                'matches_img': '/static/output/sift/custom_matches.jpg' if custom_res['matches'] > 0 else None
            }
        
        return jsonify(comparison)
        
    except Exception as e:
        return jsonify({'error': f'SIFT processing error: {str(e)}'}), 500

def run_custom_sift(img1_bgr, img2_bgr, ratio=0.75, ransac_threshold=3.0, max_matches=None):
    """Run custom SIFT implementation"""
    if not SIFT_AVAILABLE:
        return None
    
    img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    
    sift = SIFT()
    
    start = time.time()
    kps1, des1 = sift.detectAndCompute(img1_gray, None)
    time1 = time.time() - start
    
    start = time.time()
    kps2, des2 = sift.detectAndCompute(img2_gray, None)
    time2 = time.time() - start
    
    matches = match_features(kps1, des1, kps2, des2, ratio_threshold=ratio)
    
    if max_matches and len(matches) > max_matches:
        matches = matches[:max_matches]
    
    if matches:
        pts1 = np.float32([kps1[m[0]].pt for m in matches])
        pts2 = np.float32([kps2[m[1]].pt for m in matches])
    else:
        pts1 = np.empty((0, 2))
        pts2 = np.empty((0, 2))
    
    ransac = RANSAC(threshold=ransac_threshold)
    H, inlier_mask = ransac.estimate(pts1, pts2)
    inliers = int(np.sum(inlier_mask)) if inlier_mask.size > 0 else 0
    
    return {
        "keypoints1": len(kps1),
        "keypoints2": len(kps2),
        "matches": len(matches),
        "inliers": inliers,
        "kps1": kps1,
        "kps2": kps2,
        "matches_list": matches,
        "pts1": pts1,
        "pts2": pts2,
        "H": H,
        "inlier_mask": inlier_mask,
        "time_detect1": time1,
        "time_detect2": time2,
    }

def run_opencv_sift(img1_bgr, img2_bgr, ratio=0.75, ransac_threshold=3.0, max_matches=None):
    """Run OpenCV SIFT implementation"""
    sift = cv2.SIFT_create()
    
    start = time.time()
    kps1, des1 = sift.detectAndCompute(img1_bgr, None)
    time1 = time.time() - start
    
    start = time.time()
    kps2, des2 = sift.detectAndCompute(img2_bgr, None)
    time2 = time.time() - start
    
    if des1 is None or des2 is None or len(kps1) == 0 or len(kps2) == 0:
        return {
            "keypoints1": len(kps1) if kps1 is not None else 0,
            "keypoints2": len(kps2) if kps2 is not None else 0,
            "matches": 0,
            "inliers": 0,
            "kps1": kps1,
            "kps2": kps2,
            "matches_list": [],
            "pts1": np.empty((0, 2)),
            "pts2": np.empty((0, 2)),
            "H": None,
            "inlier_mask": None,
            "time_detect1": time1,
            "time_detect2": time2,
        }
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m_n in raw_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    
    good = sorted(good, key=lambda x: x.distance)
    
    if max_matches and len(good) > max_matches:
        good = good[:max_matches]
    
    if good:
        pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in good])
    else:
        pts1 = np.empty((0, 2))
        pts2 = np.empty((0, 2))
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold)
    inliers = int(mask.sum()) if mask is not None else 0
    
    return {
        "keypoints1": len(kps1),
        "keypoints2": len(kps2),
        "matches": len(good),
        "inliers": inliers,
        "kps1": kps1,
        "kps2": kps2,
        "matches_list": good,
        "pts1": pts1,
        "pts2": pts2,
        "H": H,
        "inlier_mask": mask,
        "time_detect1": time1,
        "time_detect2": time2,
    }

def visualize_keypoints(img, kps, max_keypoints=500):
    """Visualize keypoints on image"""
    vis = img.copy()
    
    for i, kp in enumerate(kps[:max_keypoints]):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size / 2)
        cv2.circle(vis, (x, y), max(size, 2), (0, 255, 0), 2)
    
    return vis

def visualize_matches(img1, img2, kps1, kps2, matches, inlier_mask=None, max_matches=100, is_opencv=False):
    """Visualize matched keypoints"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2
    
    colors = [(0, 255, 0), (255, 0, 0)]
    
    match_list = matches[:max_matches]
    
    for i, match in enumerate(match_list):
        if is_opencv:
            idx1 = match.queryIdx
            idx2 = match.trainIdx
        else:
            idx1, idx2 = match
        
        color = colors[0] if inlier_mask is None or inlier_mask[i] else colors[1]
        
        pt1 = (int(kps1[idx1].pt[0]), int(kps1[idx1].pt[1]))
        pt2 = (int(kps2[idx2].pt[0] + w1), int(kps2[idx2].pt[1]))
        
        cv2.line(vis, pt1, pt2, color, 1)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)
    
    return vis


# ============================================================================
# ASSIGNMENT 5 & 6: Object Tracking (ArUco and Traditional Trackers)
# ============================================================================

# Global variables for tracking (client-side webcam approach)
traditional_tracker = None
traditional_tracker_type = "CSRT"
tracking_bbox = None
traditional_tracker_initialized = False

@app.route('/assignment5_6')
def assignment5_6():
    """Object tracking page with ArUco and traditional trackers"""
    return render_template('assignment5_6.html')

# ArUco Marker Tracking Routes (Client-side processing)
@app.route('/assignment5_6/process_aruco', methods=['POST'])
def process_aruco_frame():
    """Process a single frame for ArUco marker detection"""
    try:
        data = request.get_json()
        img_data = data.get('image', '')
        
        # Decode base64 image
        img_data = img_data.split(',')[1] if ',' in img_data else img_data
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        parameters = aruco.DetectorParameters()
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        markers_detected = len(ids) if ids is not None else 0
        center_x, center_y = 0, 0
        
        # If markers detected
        if ids is not None and len(ids) > 0:
            # Draw detected markers
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calculate bounding box around all markers
            all_corners = []
            for corner in corners:
                for point in corner[0]:
                    all_corners.append(point)
            
            if len(all_corners) > 0:
                all_corners = np.array(all_corners)
                
                x_min = int(np.min(all_corners[:, 0]))
                y_min = int(np.min(all_corners[:, 1]))
                x_max = int(np.max(all_corners[:, 0]))
                y_max = int(np.max(all_corners[:, 1]))
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_max = min(frame.shape[0], y_max + padding)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                
                # Draw center
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Display info
                cv2.putText(frame, f"Markers: {len(ids)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Position: ({center_x}, {center_y})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No markers detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Encode processed frame
        _, buffer = cv2.imencode('.jpg', frame)
        processed_img = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': f'data:image/jpeg;base64,{processed_img}',
            'markers': markers_detected,
            'position': {'x': int(center_x), 'y': int(center_y)} if markers_detected > 0 else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Traditional Object Tracking Routes (Client-side)
@app.route('/assignment5_6/init_tracker', methods=['POST'])
def initialize_tracker():
    """Initialize tracker with ROI from client"""
    global traditional_tracker, tracking_bbox, traditional_tracker_initialized, traditional_tracker_type
    
    try:
        data = request.get_json()
        img_data = data.get('image', '')
        bbox_data = data.get('bbox', None)
        traditional_tracker_type = data.get('tracker_type', 'CSRT')
        
        # Decode base64 image
        img_data = img_data.split(',')[1] if ',' in img_data else img_data
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Use provided bbox or default center box
        if bbox_data:
            bbox = (bbox_data['x'], bbox_data['y'], bbox_data['width'], bbox_data['height'])
        else:
            h, w = frame.shape[:2]
            bbox = (w//2 - 100, h//2 - 100, 200, 200)
        
        tracking_bbox = bbox
        
        # Create tracker based on type
        if traditional_tracker_type == 'CSRT':
            traditional_tracker = cv2.TrackerCSRT_create()
        elif traditional_tracker_type == 'KCF':
            traditional_tracker = cv2.TrackerKCF_create()
        elif traditional_tracker_type == 'MOSSE':
            traditional_tracker = cv2.legacy.TrackerMOSSE_create()
        elif traditional_tracker_type == 'MIL':
            traditional_tracker = cv2.TrackerMIL_create()
        elif traditional_tracker_type == 'BOOSTING':
            traditional_tracker = cv2.legacy.TrackerBoosting_create()
        elif traditional_tracker_type == 'TLD':
            traditional_tracker = cv2.legacy.TrackerTLD_create()
        elif traditional_tracker_type == 'MEDIANFLOW':
            traditional_tracker = cv2.legacy.TrackerMedianFlow_create()
        else:
            traditional_tracker = cv2.TrackerCSRT_create()
        
        # Initialize tracker
        traditional_tracker.init(frame, bbox)
        traditional_tracker_initialized = True
        
        return jsonify({
            'status': 'success',
            'message': 'Tracker initialized',
            'bbox': {'x': bbox[0], 'y': bbox[1], 'width': bbox[2], 'height': bbox[3]}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/assignment5_6/process_traditional', methods=['POST'])
def process_traditional_frame():
    """Process a single frame with traditional tracking"""
    global traditional_tracker, traditional_tracker_initialized, tracking_bbox
    
    try:
        data = request.get_json()
        img_data = data.get('image', '')
        
        # Decode base64 image
        img_data = img_data.split(',')[1] if ',' in img_data else img_data
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        tracking_success = False
        bbox = None
        
        # If tracker is initialized, update it
        if traditional_tracker is not None and traditional_tracker_initialized:
            ret, new_bbox = traditional_tracker.update(frame)
            tracking_success = ret
            
            if ret:
                bbox = new_bbox
                # Draw bounding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
                
                cv2.putText(frame, "Tracking", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failure", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            # Draw instruction
            cv2.putText(frame, "Click 'Initialize Tracker' to start", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            
            # Draw default selection area (center box)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//2 - 100, h//2 - 100), (w//2 + 100, h//2 + 100), (255, 255, 0), 2)
        
        # Encode processed frame
        _, buffer = cv2.imencode('.jpg', frame)
        processed_img = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': f'data:image/jpeg;base64,{processed_img}',
            'tracking': tracking_success,
            'bbox': {'x': int(bbox[0]), 'y': int(bbox[1]), 'width': int(bbox[2]), 'height': int(bbox[3])} if bbox is not None else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/assignment5_6/reset_tracker', methods=['POST'])
def reset_tracker():
    """Reset the traditional tracker"""
    global traditional_tracker, traditional_tracker_initialized, tracking_bbox
    
    traditional_tracker = None
    traditional_tracker_initialized = False
    tracking_bbox = None
    
    return jsonify({'status': 'success', 'message': 'Tracker reset'})


# ============================================================================
# ASSIGNMENT 7: Real-time Pose and Hand Tracking
# ============================================================================

def initialize_camera():
    """Initialize camera and MediaPipe for Assignment 7"""
    global camera, pose, hands, mp_drawing, mp_styles
    
    if not MEDIAPIPE_AVAILABLE:
        return False
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        time.sleep(1)
        for _ in range(5):
            camera.read()
    
    if mp_drawing is None:
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
    
    if pose is None:
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    if hands is None:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    return True

def generate_frames():
    """Generate video frames for Assignment 7"""
    global output_frame, frame_idx, csv_data, is_recording
    
    if not initialize_camera():
        return
    
    prev_t = time.time()
    fps = 0.0
    
    while True:
        success, frame_bgr = camera.read()
        if not success:
            time.sleep(0.1)
            continue
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        pose_result = pose.process(frame_rgb)
        hands_result = hands.process(frame_rgb)
        
        annotated = frame_bgr.copy()
        
        if pose_result and pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                pose_result.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )
        
        if hands_result and hands_result.multi_hand_landmarks:
            for hand_lms in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    hand_lms,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                )
        
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev_t))
        prev_t = now
        
        cv2.putText(annotated, f"FPS {fps:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if is_recording:
            cv2.circle(annotated, (annotated.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(annotated, "REC", (annotated.shape[1] - 80, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if is_recording:
            with lock:
                ts_ms = int(time.time() * 1000)
                row = {"frame": frame_idx, "timestamp_ms": ts_ms}
                
                if pose_result and pose_result.pose_landmarks:
                    lms = pose_result.pose_landmarks.landmark
                    for i, lm in enumerate(lms):
                        row[f"pose_{i}_x"] = lm.x
                        row[f"pose_{i}_y"] = lm.y
                        row[f"pose_{i}_z"] = lm.z
                        row[f"pose_{i}_v"] = lm.visibility
                else:
                    for i in range(33):
                        row[f"pose_{i}_x"] = ""
                        row[f"pose_{i}_y"] = ""
                        row[f"pose_{i}_z"] = ""
                        row[f"pose_{i}_v"] = ""
                
                left = [None, None]
                right = [None, None]
                if hands_result:
                    labels = []
                    if hands_result.multi_handedness:
                        labels = [c.classification[0].label for c in hands_result.multi_handedness]
                    lms_list = hands_result.multi_hand_landmarks or []
                    for lbl, lms in zip(labels, lms_list):
                        if lbl == "Left" and left[0] is None:
                            left = [lbl, lms]
                        elif lbl == "Right" and right[0] is None:
                            right = [lbl, lms]
                
                def store_hand(hand, label):
                    if hand[1] is None:
                        for j in range(21):
                            row[f"hand_{label}_{j}_x"] = ""
                            row[f"hand_{label}_{j}_y"] = ""
                            row[f"hand_{label}_{j}_z"] = ""
                    else:
                        for j, lm in enumerate(hand[1].landmark):
                            row[f"hand_{label}_{j}_x"] = lm.x
                            row[f"hand_{label}_{j}_y"] = lm.y
                            row[f"hand_{label}_{j}_z"] = lm.z
                
                store_hand(left, "Left")
                store_hand(right, "Right")
                
                csv_data.append(row)
                frame_idx += 1
        
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/assignment7')
def assignment7():
    """Real-time pose and hand tracking page - Client-side camera (works everywhere)"""
    return render_template('assignment7.html')

@app.route('/assignment7/video_feed')
def video_feed():
    """Video feed for real-time tracking"""
    if not MEDIAPIPE_AVAILABLE:
        return jsonify({'error': 'MediaPipe not available'}), 500
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/assignment7/start_recording', methods=['POST'])
def start_recording():
    """Start recording tracking data"""
    global is_recording, csv_data, frame_idx
    with lock:
        is_recording = True
        csv_data = []
        frame_idx = 0
    return jsonify({'status': 'recording started'})

@app.route('/assignment7/stop_recording', methods=['POST'])
def stop_recording():
    """Stop recording tracking data"""
    global is_recording
    with lock:
        is_recording = False
    return jsonify({'status': 'recording stopped', 'frames': len(csv_data)})

@app.route('/assignment7/download_csv')
def download_csv():
    """Download recorded tracking data as CSV"""
    global csv_data
    
    if not csv_data:
        return jsonify({'error': 'No data recorded'}), 400
    
    output = StringIO()
    if csv_data:
        fieldnames = list(csv_data[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    output_path = Path(app.config['STATIC_FOLDER']) / 'landmarks.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        f.write(output.getvalue())
    
    return send_file(output_path, as_attachment=True, download_name='landmarks.csv')

@app.route('/assignment7/status')
def status():
    """Get recording status"""
    return jsonify({
        'is_recording': is_recording,
        'frames_recorded': len(csv_data)
    })


# ============================================================================
# UTILITY ROUTES
# ============================================================================

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files including subdirectories"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    try:
        print("\n" + "="*60)
        print("  Computer Vision Course - Unified Web Application")
        print("="*60)
        print(f"\n  Access the application at: http://localhost:5001")
        print(f"\n  Available assignments:")
        print(f"    - Assignment 1: Camera Calibration")
        print(f"    - Assignment 2: Template Matching")
        print(f"    - Assignment 3: Image Processing")
        print(f"    - Assignment 4: SIFT & Stitching")
        print(f"    - Assignment 5 & 6: Object Tracking")
        print(f"    - Assignment 7: Real-time Tracking")
        print("\n" + "="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)
    finally:
        # Assignment 7 cleanup
        if camera:
            camera.release()
        if pose:
            pose.close()
        if hands:
            hands.close()
        
        cv2.destroyAllWindows()
