document.addEventListener('DOMContentLoaded', function() {
    // Elements for number to Cistercian conversion
    const numberInput = document.getElementById('number-input');
    const convertButton = document.getElementById('convert-button');
    const cistercianResult = document.getElementById('cistercian-result');
    const conversionError = document.getElementById('conversion-error');
    const conversionLoading = document.getElementById('conversion-loading');

    // Elements for Cistercian to number recognition
    const fileInput = document.getElementById('file-input');
    const dragDropArea = document.getElementById('drag-drop-area');
    const recognizeButton = document.getElementById('recognize-button');
    const recognitionResult = document.getElementById('recognition-result');
    const recognitionError = document.getElementById('recognition-error');
    const recognitionLoading = document.getElementById('recognition-loading');
    
    // Drawing canvas elements
    const canvas = document.getElementById('drawing-canvas');
    const clearCanvasButton = document.getElementById('clear-canvas');
    const recognizeCanvasButton = document.getElementById('recognize-canvas');
    let ctx;
    
    // Initialize drawing canvas if it exists
    if (canvas) {
        ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        
        setupCanvas();
    }
    
    // Event listeners for number to Cistercian conversion
    if (convertButton) {
        convertButton.addEventListener('click', convertToCistercian);
    }
    
    if (numberInput) {
        numberInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                convertToCistercian();
            }
        });
    }
    
    // Event listeners for Cistercian to number recognition
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    if (dragDropArea) {
        dragDropArea.addEventListener('dragover', handleDragOver);
        dragDropArea.addEventListener('dragleave', handleDragLeave);
        dragDropArea.addEventListener('drop', handleDrop);
        dragDropArea.addEventListener('click', function() {
            fileInput.click();
        });
    }
    
    if (recognizeButton) {
        recognizeButton.addEventListener('click', recognizeCistercian);
    }
    
    // Canvas control event listeners
    if (clearCanvasButton) {
        clearCanvasButton.addEventListener('click', clearCanvas);
    }
    
    if (recognizeCanvasButton) {
        recognizeCanvasButton.addEventListener('click', recognizeCanvasDrawing);
    }
    
    // Function to convert number to Cistercian numeral
    function convertToCistercian() {
        const number = numberInput.value.trim();
        
        // Validate input
        if (!/^\d+$/.test(number) || parseInt(number) < 0 || parseInt(number) > 9999) {
            showError(conversionError, 'Please enter a number between 0 and 9999');
            return;
        }
        
        // Clear any previous errors
        clearError(conversionError);
        
        // Show loading indicator
        conversionLoading.style.display = 'block';
        
        // Send request to backend
        fetch('/convert-to-cistercian', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ number: parseInt(number) })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to convert number');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            conversionLoading.style.display = 'none';
            
            // Display result
            cistercianResult.innerHTML = `
                <div class="alert alert-success">
                    Number ${data.number} as Cistercian numeral:
                </div>
                <img src="${data.image}" alt="Cistercian numeral for ${data.number}" class="cistercian-image">
            `;
        })
        .catch(error => {
            // Hide loading indicator
            conversionLoading.style.display = 'none';
            
            // Show error
            showError(conversionError, error.message);
        });
    }
    
    // Function to handle file selection
    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            validateAndProcessFile(file);
        }
    }
    
    // Drag and drop handlers
    function handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        dragDropArea.classList.add('active');
    }
    
    function handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();
        dragDropArea.classList.remove('active');
    }
    
    function handleDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        dragDropArea.classList.remove('active');
        
        const file = event.dataTransfer.files[0];
        if (file) {
            validateAndProcessFile(file);
        }
    }
    
    // Validate and process the selected file
    function validateAndProcessFile(file) {
        // Check file type
        if (!file.type.match('image.*')) {
            showError(recognitionError, 'Please select an image file');
            return;
        }
        
        // Update file input display
        const fileNameDisplay = document.getElementById('file-name');
        if (fileNameDisplay) {
            fileNameDisplay.textContent = file.name;
        }
        
        // Enable recognize button
        if (recognizeButton) {
            recognizeButton.disabled = false;
        }
    }
    
    // Function to recognize Cistercian numeral from uploaded file
    function recognizeCistercian() {
        const file = fileInput.files[0];
        if (!file) {
            showError(recognitionError, 'Please select an image file first');
            return;
        }
        
        // Clear any previous errors and results
        clearError(recognitionError);
        recognitionResult.innerHTML = '';
        
        // Show loading indicator
        recognitionLoading.style.display = 'block';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send request to backend
        fetch('/recognize-cistercian', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to recognize Cistercian numeral');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            recognitionLoading.style.display = 'none';
            
            // Display result
            recognitionResult.innerHTML = `
                <div class="alert alert-success">
                    The recognized number is: <strong>${data.number}</strong>
                </div>
            `;
        })
        .catch(error => {
            // Hide loading indicator
            recognitionLoading.style.display = 'none';
            
            // Show error
            showError(recognitionError, error.message);
        });
    }
    
    // Canvas drawing functionality
    function setupCanvas() {
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        
        // Set canvas background to white
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Mouse events for drawing
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch events for mobile
        canvas.addEventListener('touchstart', handleTouchStart);
        canvas.addEventListener('touchmove', handleTouchMove);
        canvas.addEventListener('touchend', handleTouchEnd);
        
        function startDrawing(e) {
            isDrawing = true;
            [lastX, lastY] = getCoordinates(e);
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            
            const [x, y] = getCoordinates(e);
            ctx.lineTo(x, y);
            ctx.stroke();
            
            [lastX, lastY] = [x, y];
        }
        
        function stopDrawing() {
            isDrawing = false;
        }
        
        function getCoordinates(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            return [
                (e.clientX - rect.left) * scaleX,
                (e.clientY - rect.top) * scaleY
            ];
        }
        
        // Touch event handlers
        function handleTouchStart(e) {
            e.preventDefault();
            if (e.touches.length === 1) {
                const touch = e.touches[0];
                startDrawing({
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });
            }
        }
        
        function handleTouchMove(e) {
            e.preventDefault();
            if (e.touches.length === 1) {
                const touch = e.touches[0];
                draw({
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });
            }
        }
        
        function handleTouchEnd(e) {
            e.preventDefault();
            stopDrawing();
        }
    }
    
    // Clear the drawing canvas
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    // Recognize Cistercian numeral from canvas drawing
    function recognizeCanvasDrawing() {
        // Clear any previous errors and results
        clearError(recognitionError);
        recognitionResult.innerHTML = '';
        
        // Show loading indicator
        recognitionLoading.style.display = 'block';
        
        // Get canvas data as base64 image
        const imageData = canvas.toDataURL('image/png');
        
        // Create form data
        const formData = new FormData();
        formData.append('imageData', imageData);
        
        // Send request to backend
        fetch('/recognize-cistercian', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to recognize Cistercian numeral');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            recognitionLoading.style.display = 'none';
            
            // Display result
            recognitionResult.innerHTML = `
                <div class="alert alert-success">
                    The recognized number is: <strong>${data.number}</strong>
                </div>
            `;
        })
        .catch(error => {
            // Hide loading indicator
            recognitionLoading.style.display = 'none';
            
            // Show error
            showError(recognitionError, error.message);
        });
    }
    
    // Utility functions for displaying errors
    function showError(element, message) {
        element.textContent = message;
        element.style.display = 'block';
    }
    
    function clearError(element) {
        element.textContent = '';
        element.style.display = 'none';
    }
});
