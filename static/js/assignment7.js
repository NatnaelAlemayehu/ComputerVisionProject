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
        if (measuringMode) return; // Skip if in measurement mode
        
        const rect = canvasLeft.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvasLeft.width / rect.width);
        const y = (e.clientY - rect.top) * (canvasLeft.height / rect.height);
        leftPoint = {x: x, y: y};
        
        // Redraw image and mark point
        ctxLeft.drawImage(leftImage, 0, 0);
        ctxLeft.fillStyle = '#FF0000';
        ctxLeft.beginPath();
        ctxLeft.arc(x, y, 5, 0, 2 * Math.PI);
        ctxLeft.fill();
        
        document.getElementById('point-left').textContent = `Point: (${Math.round(x)}, ${Math.round(y)})`;
        
        if (leftPoint && rightPoint) {
            calculateDepth();
        }
    };
    
    canvasRight.onclick = function(e) {
        const rect = canvasRight.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvasRight.width / rect.width);
        const y = (e.clientY - rect.top) * (canvasRight.height / rect.height);
        rightPoint = {x: x, y: y};
        
        // Redraw image and mark point
        ctxRight.drawImage(rightImage, 0, 0);
        ctxRight.fillStyle = '#0000FF';
        ctxRight.beginPath();
        ctxRight.arc(x, y, 5, 0, 2 * Math.PI);
        ctxRight.fill();
        
        document.getElementById('point-right').textContent = `Point: (${Math.round(x)}, ${Math.round(y)})`;
        
        if (leftPoint && rightPoint) {
            calculateDepth();
        }
    };
}

function calculateDepth() {
    const fx = parseFloat(document.getElementById('focal-length').value);
    const baseline = parseFloat(document.getElementById('baseline').value);
    const cx = parseFloat(document.getElementById('cx').value);
    const cy = parseFloat(document.getElementById('cy').value);
    
    const disparity = leftPoint.x - rightPoint.x;
    
    if (disparity === 0) {
        document.getElementById('disparity').textContent = '0 (Invalid!)';
        document.getElementById('depth').textContent = 'Cannot calculate';
        return;
    }
    
    const depth = (fx * baseline) / disparity;
    calculatedDepth = depth;
    
    document.getElementById('disparity').textContent = disparity.toFixed(2);
    document.getElementById('depth').textContent = depth.toFixed(2) + ' mm';
    
    // Enable measurement button
    document.getElementById('measure-btn').disabled = false;
    document.getElementById('measure-instruction').textContent = 'Depth calculated! Click "Measure Distance" to measure object size.';
}

function backproject(u, v, Z, fx, cx, cy) {
    const X = (u - cx) * Z / fx;
    const Y = (v - cy) * Z / fx;  // Using fx for both like in cali.py
    return {X, Y, Z};
}

function startMeasurement() {
    if (!calculatedDepth) {
        alert('Please calculate depth first by selecting corresponding points in both images');
        return;
    }
    
    measuringMode = true;
    measurePoint1 = null;
    measurePoint2 = null;
    
    document.getElementById('measure-btn').disabled = true;
    document.getElementById('measure-instruction').textContent = 'Click two points on the LEFT image to measure distance between them';
    document.getElementById('measure-results').classList.remove('active');
    
    // Update canvas click handler for measurement
    canvasLeft.onclick = function(e) {
        if (!measuringMode) return;
        
        const rect = canvasLeft.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvasLeft.width / rect.width);
        const y = (e.clientY - rect.top) * (canvasLeft.height / rect.height);
        
        if (!measurePoint1) {
            measurePoint1 = {x: x, y: y};
            
            // Redraw everything and mark point
            ctxLeft.drawImage(leftImage, 0, 0);
            
            // Draw original depth point
            if (leftPoint) {
                ctxLeft.fillStyle = '#FF0000';
                ctxLeft.beginPath();
                ctxLeft.arc(leftPoint.x, leftPoint.y, 5, 0, 2 * Math.PI);
                ctxLeft.fill();
            }
            
            // Draw first measurement point
            ctxLeft.fillStyle = '#00FF00';
            ctxLeft.beginPath();
            ctxLeft.arc(x, y, 6, 0, 2 * Math.PI);
            ctxLeft.fill();
            
            document.getElementById('measure-instruction').textContent = 'Now click the second point';
        } else {
            measurePoint2 = {x: x, y: y};
            
            // Redraw everything and mark both points
            ctxLeft.drawImage(leftImage, 0, 0);
            
            // Draw original depth point
            if (leftPoint) {
                ctxLeft.fillStyle = '#FF0000';
                ctxLeft.beginPath();
                ctxLeft.arc(leftPoint.x, leftPoint.y, 5, 0, 2 * Math.PI);
                ctxLeft.fill();
            }
            
            // Draw measurement points
            ctxLeft.fillStyle = '#00FF00';
            ctxLeft.beginPath();
            ctxLeft.arc(measurePoint1.x, measurePoint1.y, 6, 0, 2 * Math.PI);
            ctxLeft.fill();
            
            ctxLeft.fillStyle = '#FFFF00';
            ctxLeft.beginPath();
            ctxLeft.arc(measurePoint2.x, measurePoint2.y, 6, 0, 2 * Math.PI);
            ctxLeft.fill();
            
            // Draw line between points
            ctxLeft.strokeStyle = '#FFFF00';
            ctxLeft.lineWidth = 2;
            ctxLeft.beginPath();
            ctxLeft.moveTo(measurePoint1.x, measurePoint1.y);
            ctxLeft.lineTo(measurePoint2.x, measurePoint2.y);
            ctxLeft.stroke();
            
            // Calculate distance
            calculateDistance();
            
            measuringMode = false;
            document.getElementById('measure-btn').disabled = false;
            document.getElementById('measure-instruction').textContent = 'Click "Measure Distance" again to measure another object';
        }
    };
}

function calculateDistance() {
    const fx = parseFloat(document.getElementById('focal-length').value);
    const cx = parseFloat(document.getElementById('cx').value);
    const cy = parseFloat(document.getElementById('cy').value);
    
    // Backproject both points using the calculated depth
    const point1_3d = backproject(measurePoint1.x, measurePoint1.y, calculatedDepth, fx, cx, cy);
    const point2_3d = backproject(measurePoint2.x, measurePoint2.y, calculatedDepth, fx, cx, cy);
    
    // Calculate Euclidean distance
    const distance = Math.sqrt(
        Math.pow(point1_3d.X - point2_3d.X, 2) +
        Math.pow(point1_3d.Y - point2_3d.Y, 2) +
        Math.pow(point1_3d.Z - point2_3d.Z, 2)
    );
    
    // Display results
    document.getElementById('point1-coords').textContent = 
        `Point 1: (${Math.round(measurePoint1.x)}, ${Math.round(measurePoint1.y)})`;
    document.getElementById('point2-coords').textContent = 
        `Point 2: (${Math.round(measurePoint2.x)}, ${Math.round(measurePoint2.y)})`;
    document.getElementById('measured-distance').textContent = 
        `${distance.toFixed(2)} mm`;
    
    document.getElementById('measure-results').classList.add('active');
}

// Tracking Functions
let isRecording = false;
let statusInterval = null;

async function startRecording() {
    try {
        const res = await fetch('/assignment7/start_recording', { method: 'POST' });
        const data = await res.json();
        isRecording = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('statusText').innerHTML = '<span class="recording-indicator"></span>Recording...';
        statusInterval = setInterval(updateStatus, 1000);
    } catch(e) {
        alert('Error starting recording: ' + e.message);
    }
}

async function stopRecording() {
    try {
        const res = await fetch('/assignment7/stop_recording', { method: 'POST' });
        const data = await res.json();
        isRecording = false;
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('downloadBtn').disabled = false;
        document.getElementById('statusText').textContent = 'Recording stopped';
        if(statusInterval) clearInterval(statusInterval);
    } catch(e) {
        alert('Error stopping recording: ' + e.message);
    }
}

async function downloadCSV() {
    window.location.href = '/assignment7/download_csv';
}

async function updateStatus() {
    try {
        const res = await fetch('/assignment7/status');
        const data = await res.json();
        document.getElementById('frameCount').textContent = `Frames recorded: ${data.frames_recorded}`;
    } catch(e) {
        console.error('Error updating status:', e);
    }
}

function showLoading(type) { document.getElementById(`${type}-loading`).classList.add('active'); }
function hideLoading(type) { document.getElementById(`${type}-loading`).classList.remove('active'); }
function showResults(type) { document.getElementById(`${type}-results`).classList.add('active'); }
function hideResults(type) { document.getElementById(`${type}-results`).classList.remove('active'); }
function showError(type, message) { const errorDiv = document.getElementById(`${type}-error`); errorDiv.textContent = message; errorDiv.classList.add('active'); }
function hideError(type) { document.getElementById(`${type}-error`).classList.remove('active'); }
