function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById(tabName).classList.add('active');
}

function previewFile(type) {
    // Map type to the correct input/button prefix
    const typeMap = {
        'gradients': 'gradient',
        'edges': 'edge',
        'corners': 'corner',
        'contours': 'contour',
        'aruco': 'aruco'
    };
    
    const prefix = typeMap[type] || type;
    const fileInput = document.getElementById(`${prefix}-file`);
    const filenameDisplay = document.getElementById(`${prefix}-filename`);
    const processBtn = document.getElementById(`${prefix}-process-btn`);
    
    if (fileInput.files.length > 0) {
        filenameDisplay.textContent = `Selected: ${fileInput.files[0].name}`;
        processBtn.disabled = false;
    }
}

function updateThreshold() {
    const threshold = document.getElementById('edge-threshold').value;
    document.getElementById('threshold-value').textContent = threshold;
}

async function processGradients() {
    const fileInput = document.getElementById('gradient-file');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    showLoading('gradient');
    hideError('gradient');
    hideResults('gradient');
    try {
        const response = await fetch('/assignment3/process_gradients', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Processing failed');
        document.getElementById('gradient-original').src = data.original + '?' + new Date().getTime();
        document.getElementById('gradient-magnitude').src = data.magnitude + '?' + new Date().getTime();
        document.getElementById('gradient-angle').src = data.angle + '?' + new Date().getTime();
        document.getElementById('gradient-log').src = data.log + '?' + new Date().getTime();
        hideLoading('gradient');
        showResults('gradient');
    } catch (error) {
        hideLoading('gradient');
        showError('gradient', error.message);
    }
}

async function processEdges() {
    const fileInput = document.getElementById('edge-file');
    const threshold = document.getElementById('edge-threshold').value;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('threshold', threshold);
    showLoading('edge');
    hideError('edge');
    hideResults('edge');
    try {
        const response = await fetch('/assignment3/process_edges', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Processing failed');
        document.getElementById('edge-original').src = data.original + '?' + new Date().getTime();
        document.getElementById('edge-edges').src = data.edges + '?' + new Date().getTime();
        hideLoading('edge');
        showResults('edge');
    } catch (error) {
        hideLoading('edge');
        showError('edge', error.message);
    }
}

async function processCorners() {
    const fileInput = document.getElementById('corner-file');
    const maxCorners = document.getElementById('max-corners').value;
    const qualityLevel = document.getElementById('quality-level').value;
    const minDistance = document.getElementById('min-distance').value;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('maxCorners', maxCorners);
    formData.append('qualityLevel', qualityLevel);
    formData.append('minDistance', minDistance);
    showLoading('corner');
    hideError('corner');
    hideResults('corner');
    try {
        const response = await fetch('/assignment3/process_corners', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Processing failed');
        document.getElementById('corner-original').src = data.original + '?' + new Date().getTime();
        document.getElementById('corner-corners').src = data.corners + '?' + new Date().getTime();
        document.getElementById('corner-count').textContent = `Detected ${data.count} corners`;
        hideLoading('corner');
        showResults('corner');
    } catch (error) {
        hideLoading('corner');
        showError('corner', error.message);
    }
}

async function processContours() {
    const fileInput = document.getElementById('contour-file');
    const method = document.getElementById('contour-method').value;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('method', method);
    showLoading('contour');
    hideError('contour');
    hideResults('contour');
    try {
        const response = await fetch('/assignment3/process_contours', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Processing failed');
        document.getElementById('contour-original').src = data.original + '?' + new Date().getTime();
        document.getElementById('contour-binary').src = data.binary + '?' + new Date().getTime();
        document.getElementById('contour-contours').src = data.contours + '?' + new Date().getTime();
        document.getElementById('contour-count').textContent = `Detected ${data.count} contours`;
        hideLoading('contour');
        showResults('contour');
    } catch (error) {
        hideLoading('contour');
        showError('contour', error.message);
    }
}

async function processAruco() {
    const fileInput = document.getElementById('aruco-file');
    const buffer = document.getElementById('buffer-distance').value;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('buffer', buffer);
    showLoading('aruco');
    hideError('aruco');
    hideResults('aruco');
    try {
        const response = await fetch('/assignment3/process_aruco', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Processing failed');
        document.getElementById('aruco-original').src = data.original + '?' + new Date().getTime();
        document.getElementById('aruco-result').src = data.result + '?' + new Date().getTime();
        document.getElementById('aruco-info').textContent = `Detected ${data.markers_detected} ArUco markers (IDs: ${data.marker_ids.join(', ')})`;
        hideLoading('aruco');
        showResults('aruco');
    } catch (error) {
        hideLoading('aruco');
        showError('aruco', error.message);
    }
}

function displayComparison() {
    hideError('comparison');
    hideResults('comparison');
    
    try {
        // Get all image filenames from inputs and construct paths
        const groupFolders = ['g1', 'g2', 'g3'];
        
        for (let group = 1; group <= 3; group++) {
            for (let img = 1; img <= 3; img++) {
                const input = document.getElementById(`comp-group${group}-${img}`);
                const filename = input.value.trim() || `${img}.jpg`;
                const imgElement = document.getElementById(`comp-img-${group}-${img}`);
                
                if (imgElement) {
                    // Construct path: uploads/assignment3/question5/g1/1.jpg
                    const path = `/uploads/assignment3/question5/${groupFolders[group-1]}/${filename}`;
                    imgElement.src = path + '?' + new Date().getTime();
                    
                    imgElement.onerror = function() {
                        this.style.display = 'none';
                        const header = this.parentElement.querySelector('h4');
                        if (!header.textContent.includes('Not Found')) {
                            header.textContent += ' (Not Found)';
                        }
                    };
                    imgElement.onload = function() {
                        this.style.display = 'block';
                        const header = this.parentElement.querySelector('h4');
                        header.textContent = header.textContent.replace(' (Not Found)', '');
                    };
                }
            }
        }
        
        showResults('comparison');
        
    } catch (error) {
        showError('comparison', error.message);
    }
}

function showLoading(type) { document.getElementById(`${type}-loading`).classList.add('active'); }
function hideLoading(type) { document.getElementById(`${type}-loading`).classList.remove('active'); }
function showResults(type) { document.getElementById(`${type}-results`).classList.add('active'); }
function hideResults(type) { document.getElementById(`${type}-results`).classList.remove('active'); }
function showError(type, message) { const errorDiv = document.getElementById(`${type}-error`); errorDiv.textContent = message; errorDiv.classList.add('active'); }
function hideError(type) { document.getElementById(`${type}-error`).classList.remove('active'); }
