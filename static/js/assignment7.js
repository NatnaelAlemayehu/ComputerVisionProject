// ========================================
// STEREO VISION (unchanged)
// ========================================
function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById(tabName).classList.add('active');
}

// Stereo Vision Variables
let leftImage = null;
let rightImage = null;
let leftPoint = null;
let rightPoint = null;
let canvasLeft = null;
let canvasRight = null;
let ctxLeft = null;
let ctxRight = null;
let calculatedDepth = null;
let measurePoint1 = null;
let measurePoint2 = null;
let measuringMode = false;

function previewFile(type, imageNum) {
    const fileInput = document.getElementById(`stereo-img${imageNum}`);
    const filenameDisplay = document.getElementById(`stereo-filename${imageNum}`);
    const processBtn = document.getElementById('stereo-process-btn');
    
    if (fileInput.files.length > 0) {
        filenameDisplay.textContent = `Selected: ${fileInput.files[0].name}`;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            if (imageNum === 1) {
                leftImage = new Image();
                leftImage.src = e.target.result;
            } else {
                rightImage = new Image();
                rightImage.src = e.target.result;
            }
        };
        reader.readAsDataURL(fileInput.files[0]);
        
        // Enable button if both images are selected
        const file1 = document.getElementById('stereo-img1').files.length > 0;
        const file2 = document.getElementById('stereo-img2').files.length > 0;
        processBtn.disabled = !(file1 && file2);
    }
}

function processStereo() {
    if (!leftImage || !rightImage) {
        showError('stereo', 'Please select both images');
        return;
    }
    
    hideError('stereo');
    showResults('stereo');
    
    // Setup canvases
    canvasLeft = document.getElementById('canvas-left');
    canvasRight = document.getElementById('canvas-right');
    ctxLeft = canvasLeft.getContext('2d');
    ctxRight = canvasRight.getContext('2d');
    
    // Wait for images to load
    leftImage.onload = function() {
        canvasLeft.width = leftImage.width;
        canvasLeft.height = leftImage.height;
        ctxLeft.drawImage(leftImage, 0, 0);
    };
    
    rightImage.onload = function() {
        canvasRight.width = rightImage.width;
        canvasRight.height = rightImage.height;
        ctxRight.drawImage(rightImage, 0, 0);
    };
    
    // Trigger onload if already loaded
    if (leftImage.complete) {
        canvasLeft.width = leftImage.width;
        canvasLeft.height = leftImage.height;
        ctxLeft.drawImage(leftImage, 0, 0);
    }
    
    if (rightImage.complete) {
        canvasRight.width = rightImage.width;
        canvasRight.height = rightImage.height;
        ctxRight.drawImage(rightImage, 0, 0);
    }
    
    // Add click handlers
    setupDepthClickHandlers();
}

function setupDepthClickHandlers() {
    canvasLeft.onclick = function(e) {
        if (measuringMode) {
            handleMeasurementClick(e);
            return;
        }
        
        const rect = canvasLeft.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvasLeft.width / rect.width);
        const y = (e.clientY - rect.top) * (canvasLeft.height / rect.height);
        
        leftPoint = {x: x, y: y};
        ctxLeft.drawImage(leftImage, 0, 0);
        ctxLeft.fillStyle = '#FF0000';
        ctxLeft.beginPath();
        ctxLeft.arc(x, y, 5, 0, 2 * Math.PI);
        ctxLeft.fill();
        
        document.getElementById('point-left').textContent = `Selected: (${Math.round(x)}, ${Math.round(y)})`;
        checkAndCalculate();
    };
    
    canvasRight.onclick = function(e) {
        const rect = canvasRight.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvasRight.width / rect.width);
        const y = (e.clientY - rect.top) * (canvasRight.height / rect.height);
        
        rightPoint = {x: x, y: y};
        ctxRight.drawImage(rightImage, 0, 0);
        ctxRight.fillStyle = '#00FF00';
        ctxRight.beginPath();
        ctxRight.arc(x, y, 5, 0, 2 * Math.PI);
        ctxRight.fill();
        
        document.getElementById('point-right').textContent = `Selected: (${Math.round(x)}, ${Math.round(y)})`;
        checkAndCalculate();
    };
}

function checkAndCalculate() {
    if (leftPoint && rightPoint) {
        const disparity = Math.abs(leftPoint.x - rightPoint.x);
        const fx = parseFloat(document.getElementById('focal-length').value);
        const baseline = parseFloat(document.getElementById('baseline').value);
        
        if (disparity === 0) {
            document.getElementById('disparity').textContent = 'Invalid (zero disparity)';
            document.getElementById('depth').textContent = '-';
            return;
        }
        
        calculatedDepth = (fx * baseline) / disparity;
        
        document.getElementById('disparity').textContent = disparity.toFixed(2);
        document.getElementById('depth').textContent = `${calculatedDepth.toFixed(2)} mm`;
        document.getElementById('stereo-info').classList.add('active');
        document.getElementById('measure-btn').disabled = false;
    }
}

function startMeasurement() {
    measuringMode = true;
    measurePoint1 = null;
    measurePoint2 = null;
    document.getElementById('measure-btn').disabled = true;
    document.getElementById('measure-instruction').textContent = 'Click two points in the left image to measure distance';
    document.getElementById('measure-results').classList.remove('active');
}

function handleMeasurementClick(e) {
    const rect = canvasLeft.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasLeft.width / rect.width);
    const y = (e.clientY - rect.top) * (canvasLeft.height / rect.height);
    
    if (!measurePoint1) {
        measurePoint1 = {x: x, y: y};
        
        ctxLeft.drawImage(leftImage, 0, 0);
        if (leftPoint) {
            ctxLeft.fillStyle = '#FF0000';
            ctxLeft.beginPath();
            ctxLeft.arc(leftPoint.x, leftPoint.y, 5, 0, 2 * Math.PI);
            ctxLeft.fill();
        }
        
        ctxLeft.fillStyle = '#00FF00';
        ctxLeft.beginPath();
        ctxLeft.arc(x, y, 6, 0, 2 * Math.PI);
        ctxLeft.fill();
        
        document.getElementById('measure-instruction').textContent = 'Now click the second point';
    } else {
        measurePoint2 = {x: x, y: y};
        
        ctxLeft.drawImage(leftImage, 0, 0);
        if (leftPoint) {
            ctxLeft.fillStyle = '#FF0000';
            ctxLeft.beginPath();
            ctxLeft.arc(leftPoint.x, leftPoint.y, 5, 0, 2 * Math.PI);
            ctxLeft.fill();
        }
        
        ctxLeft.fillStyle = '#00FF00';
        ctxLeft.beginPath();
        ctxLeft.arc(measurePoint1.x, measurePoint1.y, 6, 0, 2 * Math.PI);
        ctxLeft.fill();
        
        ctxLeft.fillStyle = '#FFFF00';
        ctxLeft.beginPath();
        ctxLeft.arc(measurePoint2.x, measurePoint2.y, 6, 0, 2 * Math.PI);
        ctxLeft.fill();
        
        ctxLeft.strokeStyle = '#FFFF00';
        ctxLeft.lineWidth = 2;
        ctxLeft.beginPath();
        ctxLeft.moveTo(measurePoint1.x, measurePoint1.y);
        ctxLeft.lineTo(measurePoint2.x, measurePoint2.y);
        ctxLeft.stroke();
        
        calculateDistance();
        
        measuringMode = false;
        document.getElementById('measure-btn').disabled = false;
        document.getElementById('measure-instruction').textContent = 'Click "Measure Distance" again to measure another object';
    }
}

function backproject(u, v, Z, fx, cx, cy) {
    const X = (u - cx) * Z / fx;
    const Y = (v - cy) * Z / fx;
    return {X: X, Y: Y, Z: Z};
}

function calculateDistance() {
    const fx = parseFloat(document.getElementById('focal-length').value);
    const cx = parseFloat(document.getElementById('cx').value);
    const cy = parseFloat(document.getElementById('cy').value);
    
    const point1_3d = backproject(measurePoint1.x, measurePoint1.y, calculatedDepth, fx, cx, cy);
    const point2_3d = backproject(measurePoint2.x, measurePoint2.y, calculatedDepth, fx, cx, cy);
    
    const distance = Math.sqrt(
        Math.pow(point1_3d.X - point2_3d.X, 2) +
        Math.pow(point1_3d.Y - point2_3d.Y, 2) +
        Math.pow(point1_3d.Z - point2_3d.Z, 2)
    );
    
    document.getElementById('point1-coords').textContent = 
        `Point 1: (${Math.round(measurePoint1.x)}, ${Math.round(measurePoint1.y)})`;
    document.getElementById('point2-coords').textContent = 
        `Point 2: (${Math.round(measurePoint2.x)}, ${Math.round(measurePoint2.y)})`;
    document.getElementById('measured-distance').textContent = 
        `${distance.toFixed(2)} mm`;
    
    document.getElementById('measure-results').classList.add('active');
}

// ========================================
// CLIENT-SIDE POSE TRACKING WITH MEDIAPIPE
// ========================================

let videoElement = null;
let canvasElement = null;
let canvasCtx = null;
let pose = null;
let hands = null;
let cameraStarted = false;
let isRecording = false;
let recordedData = [];
let frameCount = 0;
let animationId = null;
let lastFrameTime = Date.now();
let fps = 0;

// MediaPipe drawing utilities
const drawConnectors = window.drawConnectors;
const drawLandmarks = window.drawLandmarks;

async function startCamera() {
    try {
        videoElement = document.getElementById('webcam');
        canvasElement = document.getElementById('output-canvas');
        canvasCtx = canvasElement.getContext('2d');
        
        document.getElementById('statusText').textContent = 'Initializing camera...';
        
        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
        
        videoElement.srcObject = stream;
        
        // Wait for video to load
        await new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                resolve();
            };
        });
        
        document.getElementById('statusText').textContent = 'Initializing MediaPipe...';
        
        // Initialize MediaPipe Pose
        pose = new Pose({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
            }
        });
        
        pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        // Initialize MediaPipe Hands
        hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });
        
        hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        // Setup result handlers
        pose.onResults(onPoseResults);
        hands.onResults(onHandsResults);
        
        cameraStarted = true;
        document.getElementById('startCameraBtn').disabled = true;
        document.getElementById('startRecBtn').disabled = false;
        document.getElementById('statusText').textContent = 'Camera ready! Click "Start Recording" to begin.';
        
        // Start processing frames
        processFrame();
        
    } catch (error) {
        console.error('Error starting camera:', error);
        document.getElementById('statusText').textContent = 'Error: ' + error.message;
        alert('Failed to start camera: ' + error.message + '\n\nPlease ensure:\n1. Camera permissions are granted\n2. You\'re using HTTPS or localhost\n3. Your browser supports WebRTC');
    }
}

let currentPoseResults = null;
let currentHandsResults = null;

async function processFrame() {
    if (!cameraStarted) return;
    
    // Send frame to both pose and hands
    await pose.send({image: videoElement});
    await hands.send({image: videoElement});
    
    // Request next frame
    animationId = requestAnimationFrame(processFrame);
}

function onPoseResults(results) {
    currentPoseResults = results;
    drawResults();
}

function onHandsResults(results) {
    currentHandsResults = results;
    drawResults();
}

function drawResults() {
    if (!canvasCtx) return;
    
    // Calculate FPS
    const now = Date.now();
    const delta = (now - lastFrameTime) / 1000;
    fps = fps * 0.9 + (1 / delta) * 0.1;
    lastFrameTime = now;
    
    // Clear canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw video frame
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    
    // Draw pose landmarks
    if (currentPoseResults && currentPoseResults.poseLandmarks) {
        drawConnectors(canvasCtx, currentPoseResults.poseLandmarks, window.POSE_CONNECTIONS,
                      {color: '#00FF00', lineWidth: 4});
        drawLandmarks(canvasCtx, currentPoseResults.poseLandmarks,
                     {color: '#FF0000', lineWidth: 2});
    }
    
    // Draw hand landmarks
    if (currentHandsResults && currentHandsResults.multiHandLandmarks) {
        for (const landmarks of currentHandsResults.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS,
                         {color: '#00CC00', lineWidth: 5});
            drawLandmarks(canvasCtx, landmarks,
                        {color: '#FF0000', fillColor: '#00FF00'});
        }
    }
    
    // Draw FPS
    canvasCtx.font = '24px Arial';
    canvasCtx.fillStyle = '#00FF00';
    canvasCtx.fillText(`FPS: ${fps.toFixed(1)}`, 10, 30);
    
    // Draw recording indicator
    if (isRecording) {
        canvasCtx.fillStyle = '#FF0000';
        canvasCtx.beginPath();
        canvasCtx.arc(canvasElement.width - 30, 30, 10, 0, 2 * Math.PI);
        canvasCtx.fill();
        canvasCtx.fillText('REC', canvasElement.width - 80, 38);
    }
    
    canvasCtx.restore();
    
    // Record data if recording
    if (isRecording) {
        recordFrame();
    }
    
    // Update FPS display
    document.getElementById('fpsDisplay').textContent = `FPS: ${fps.toFixed(1)}`;
}

function recordFrame() {
    const timestamp = Date.now();
    const row = {
        frame: frameCount,
        timestamp_ms: timestamp
    };
    
    // Record pose landmarks
    if (currentPoseResults && currentPoseResults.poseLandmarks) {
        currentPoseResults.poseLandmarks.forEach((landmark, i) => {
            row[`pose_${i}_x`] = landmark.x;
            row[`pose_${i}_y`] = landmark.y;
            row[`pose_${i}_z`] = landmark.z;
            row[`pose_${i}_v`] = landmark.visibility || 1.0;
        });
    } else {
        // Empty pose data
        for (let i = 0; i < 33; i++) {
            row[`pose_${i}_x`] = '';
            row[`pose_${i}_y`] = '';
            row[`pose_${i}_z`] = '';
            row[`pose_${i}_v`] = '';
        }
    }
    
    // Record hand landmarks
    const leftHand = { landmarks: null };
    const rightHand = { landmarks: null };
    
    if (currentHandsResults && currentHandsResults.multiHandLandmarks) {
        currentHandsResults.multiHandedness.forEach((handedness, idx) => {
            const label = handedness.label; // "Left" or "Right"
            const landmarks = currentHandsResults.multiHandLandmarks[idx];
            
            if (label === 'Left' && !leftHand.landmarks) {
                leftHand.landmarks = landmarks;
            } else if (label === 'Right' && !rightHand.landmarks) {
                rightHand.landmarks = landmarks;
            }
        });
    }
    
    // Store left hand
    if (leftHand.landmarks) {
        leftHand.landmarks.forEach((landmark, j) => {
            row[`hand_Left_${j}_x`] = landmark.x;
            row[`hand_Left_${j}_y`] = landmark.y;
            row[`hand_Left_${j}_z`] = landmark.z;
        });
    } else {
        for (let j = 0; j < 21; j++) {
            row[`hand_Left_${j}_x`] = '';
            row[`hand_Left_${j}_y`] = '';
            row[`hand_Left_${j}_z`] = '';
        }
    }
    
    // Store right hand
    if (rightHand.landmarks) {
        rightHand.landmarks.forEach((landmark, j) => {
            row[`hand_Right_${j}_x`] = landmark.x;
            row[`hand_Right_${j}_y`] = landmark.y;
            row[`hand_Right_${j}_z`] = landmark.z;
        });
    } else {
        for (let j = 0; j < 21; j++) {
            row[`hand_Right_${j}_x`] = '';
            row[`hand_Right_${j}_y`] = '';
            row[`hand_Right_${j}_z`] = '';
        }
    }
    
    recordedData.push(row);
    frameCount++;
    document.getElementById('frameCount').textContent = `Frames recorded: ${frameCount}`;
}

async function startRecording() {
    if (!cameraStarted) {
        alert('Please start the camera first!');
        return;
    }
    
    isRecording = true;
    recordedData = [];
    frameCount = 0;
    
    document.getElementById('startRecBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('statusText').innerHTML = '<span style="color: red;">●</span> Recording...';
}

async function stopRecording() {
    isRecording = false;
    
    document.getElementById('startRecBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('downloadBtn').disabled = false;
    document.getElementById('statusText').textContent = `Recording stopped. ${frameCount} frames captured.`;
}

function downloadCSV() {
    if (recordedData.length === 0) {
        alert('No data to download!');
        return;
    }
    
    // Convert to CSV
    const headers = Object.keys(recordedData[0]);
    let csv = headers.join(',') + '\n';
    
    recordedData.forEach(row => {
        const values = headers.map(header => {
            const value = row[header];
            return value === '' ? '' : value;
        });
        csv += values.join(',') + '\n';
    });
    
    // Create download link
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `landmarks_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    document.getElementById('statusText').textContent = 'CSV downloaded successfully!';
}

// Helper functions
function showLoading(type) { 
    document.getElementById(`${type}-loading`).classList.add('active'); 
}

function hideLoading(type) { 
    document.getElementById(`${type}-loading`).classList.remove('active'); 
}

function showResults(type) { 
    document.getElementById(`${type}-results`).classList.add('active'); 
}

function hideResults(type) { 
    document.getElementById(`${type}-results`).classList.remove('active'); 
}

function showError(type, message) { 
    const errorDiv = document.getElementById(`${type}-error`); 
    errorDiv.textContent = message; 
    errorDiv.classList.add('active'); 
}

function hideError(type) { 
    document.getElementById(`${type}-error`).classList.remove('active'); 
}
