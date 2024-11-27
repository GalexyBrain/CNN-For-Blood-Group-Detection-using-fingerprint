// Handle file selection and preview
document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];

    // Check if a file is selected
    if (file) {
        // Display file name
        const fileName = document.createElement('p');
        fileName.textContent = `Selected File: ${file.name}`;
        fileName.style.color = '#fff';
        fileName.style.marginTop = '10px';

        // Remove any previous file info or image preview
        const existingPreview = document.querySelector('.file-preview');
        if (existingPreview) existingPreview.remove();

        // Create a container for preview
        const previewContainer = document.createElement('div');
        previewContainer.classList.add('file-preview');
        previewContainer.style.marginTop = '10px';

        // Add file name to preview container
        previewContainer.appendChild(fileName);

        // Create an image element for preview
        const imgPreview = document.createElement('img');
        imgPreview.style.maxWidth = '100%';
        imgPreview.style.maxHeight = '200px';
        imgPreview.style.border = '2px solid #fff';
        imgPreview.style.borderRadius = '10px';
        imgPreview.style.marginTop = '10px';

        // Load the image file
        const reader = new FileReader();
        reader.onload = function(e) {
            imgPreview.src = e.target.result; // Set the image source to the file
        };
        reader.readAsDataURL(file);

        // Add the image preview to the container
        previewContainer.appendChild(imgPreview);

        // Insert the preview container after the file input
        event.target.insertAdjacentElement('afterend', previewContainer);
    }
});

// Handle form submission for prediction
document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent the form from reloading the page

    // Clear previous results and error messages
    document.getElementById('result').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');
    document.getElementById('bloodGroup').textContent = '';
    document.getElementById('errorMessage').textContent = '';

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload a file!");
        return;
    }

    // Show acknowledgment message when uploading
    const uploadAcknowledgment = document.createElement('p');
    uploadAcknowledgment.textContent = 'Processing image... Please wait.';
    uploadAcknowledgment.style.color = '#ffeb3b'; // Yellow color for acknowledgment
    uploadAcknowledgment.style.marginTop = '10px';
    uploadAcknowledgment.style.fontWeight = 'bold';

    // Remove any previous acknowledgment
    const existingAck = document.querySelector('.upload-ack');
    if (existingAck) existingAck.remove();

    // Insert acknowledgment below the form
    document.getElementById('uploadForm').insertAdjacentElement('afterend', uploadAcknowledgment);
    uploadAcknowledgment.classList.add('upload-ack');

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Make an API call to the Flask server
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            // Remove the acknowledgment message
            uploadAcknowledgment.remove();
            
            // Display the predicted blood group
            document.getElementById('bloodGroup').textContent = result.blood_group;
            document.getElementById('result').classList.remove('hidden');
        } else {
            // Remove the acknowledgment message
            uploadAcknowledgment.remove();

            // Display the error message
            document.getElementById('errorMessage').textContent = result.error || 'Unknown error occurred.';
            document.getElementById('error').classList.remove('hidden');
        }
    } catch (error) {
        // Remove the acknowledgment message
        uploadAcknowledgment.remove();

        // Handle network errors
        document.getElementById('errorMessage').textContent = 'Failed to connect to the server.';
        document.getElementById('error').classList.remove('hidden');
    }
});
