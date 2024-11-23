document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const errorElement = document.getElementById('error');
    const resultElement = document.getElementById('result');
    const bloodGroupElement = document.getElementById('bloodGroup');
    const uploadStatus = document.createElement('p'); // Status message for file upload
    uploadStatus.className = 'upload-status';
    fileInput.parentNode.insertBefore(uploadStatus, fileInput.nextSibling);

    // Handle file input change
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            uploadStatus.textContent = `File "${fileInput.files[0].name}" is selected.`;
            uploadStatus.style.color = 'green';
        } else {
            uploadStatus.textContent = '';
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent default form submission

        // Clear previous error messages and keep the result section visible
        errorElement.textContent = '';
        bloodGroupElement.textContent = '';
        resultElement.style.display = 'none';

        // Validate file input
        if (!fileInput.files.length) {
            errorElement.textContent = 'Please select an image to upload.';
            return;
        }

        // Show uploading status
        uploadStatus.textContent = 'Uploading file... Please wait.';
        uploadStatus.style.color = 'orange';

        // Prepare the file for upload using FormData
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            // Make a POST request to the server
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData,
            });

            // Parse the JSON response
            const data = await response.json();

            if (response.ok) {
                // Update the upload status and display the detected blood group
                uploadStatus.textContent = 'File uploaded successfully!';
                uploadStatus.style.color = 'green';

                bloodGroupElement.textContent = `Your Blood Group: ${data.blood_group}`;
                resultElement.style.display = 'block';
            } else {
                // Handle server errors
                throw new Error(data.error || 'Failed to detect blood group.');
            }
        } catch (error) {
            // Display error messages to the user
            uploadStatus.textContent = 'Upload failed. Please try again.';
            uploadStatus.style.color = 'red';
            errorElement.textContent = `Error: ${error.message}`;
        }
    });
});
