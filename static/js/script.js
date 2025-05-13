document.addEventListener('DOMContentLoaded', function() {
    const numberInput = document.getElementById('number-input');
    const convertButton = document.getElementById('convert-button');
    const cistercianResult = document.getElementById('cistercian-result');
    const conversionError = document.getElementById('conversion-error');
    const conversionLoading = document.getElementById('conversion-loading');

    const fileInput = document.getElementById('file-input');
    const dragDropArea = document.getElementById('drag-drop-area');
    const recognizeButton = document.getElementById('recognize-button');
    const recognitionResult = document.getElementById('recognition-result');
    const recognitionError = document.getElementById('recognition-error');
    const recognitionLoading = document.getElementById('recognition-loading');
    
    const canvas = document.getElementById('drawing-canvas');
    const clearCanvasButton = document.getElementById('clear-canvas');
    const recognizeCanvasButton = document.getElementById('recognize-canvas');
    let ctx;

    if (canvas) {
        ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        
        setupCanvas();
    }
    
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
    
    if (clearCanvasButton) {
        clearCanvasButton.addEventListener('click', clearCanvas);
    }
    
    if (recognizeCanvasButton) {
        recognizeCanvasButton.addEventListener('click', recognizeCanvasDrawing);
    }
    
    function convertToCistercian() {
        const number = numberInput.value.trim();
        
        if (!/^\d+$/.test(number) || parseInt(number) < 0 || parseInt(number) > 9999) {
            showError(conversionError, 'Please enter a number between 0 and 9999');
            return;
        }

        clearError(conversionError);

        conversionLoading.style.display = 'block';

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
            conversionLoading.style.display = 'none';

            cistercianResult.innerHTML = `
                <div class="alert alert-success">
                    Number ${data.number} as Cistercian numeral:
                </div>
                <img src="${data.image}" alt="Cistercian numeral for ${data.number}" class="cistercian-image">
            `;
        })
        .catch(error => {
            conversionLoading.style.display = 'none';

            showError(conversionError, error.message);
        });
    }

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            validateAndProcessFile(file);
        }
    }

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

    function validateAndProcessFile(file) {
        if (!file.type.match('image.*')) {
            showError(recognitionError, 'Please select an image file');
            return;
        }

        const fileNameDisplay = document.getElementById('file-name');
        if (fileNameDisplay) {
            fileNameDisplay.textContent = file.name;
        }

        if (recognizeButton) {
            recognizeButton.disabled = false;
        }
    }

    function recognizeCistercian() {
        const file = fileInput.files[0];
        if (!file) {
            showError(recognitionError, 'Please select an image file first');
            return;
        }

        clearError(recognitionError);
        recognitionResult.innerHTML = '';

        recognitionLoading.style.display = 'block';

        const formData = new FormData();
        formData.append('file', file);

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
            recognitionLoading.style.display = 'none';

            recognitionResult.innerHTML = `
                <div class="alert alert-success">
                    The recognized number is: <strong>${data.number}</strong>
                </div>
            `;
        })
        .catch(error => {
            recognitionLoading.style.display = 'none';
            
            showError(recognitionError, error.message);
        });
    }

    function setupCanvas() {
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

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

    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    function recognizeCanvasDrawing() {
        clearError(recognitionError);
        recognitionResult.innerHTML = '';

        recognitionLoading.style.display = 'block';
    
        const imageData = canvas.toDataURL('image/png');

        const formData = new FormData();
        formData.append('imageData', imageData);

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
            recognitionLoading.style.display = 'none';

            recognitionResult.innerHTML = `
                <div class="alert alert-success">
                    The recognized number is: <strong>${data.number}</strong>
                </div>
            `;
        })
        .catch(error => {
            recognitionLoading.style.display = 'none';

            showError(recognitionError, error.message);
        });
    }

    function showError(element, message) {
        element.textContent = message;
        element.style.display = 'block';
    }
    
    function clearError(element) {
        element.textContent = '';
        element.style.display = 'none';
    }
});
