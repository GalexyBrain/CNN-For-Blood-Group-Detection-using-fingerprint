// JavaScript for handling form submission and displaying results
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const errorElement = document.getElementById('error');
    const resultElement = document.getElementById('result');
    const bloodGroupElement = document.getElementById('bloodGroup');

    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent default form submission

        // Clear previous error messages and hide the result section
        errorElement.textContent = '';
        resultElement.style.display = 'none';

        // Validate file input
        if (!fileInput.files.length) {
            errorElement.textContent = 'Please select an image to upload.';
            return;
        }

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
                // Display the detected blood group
                bloodGroupElement.textContent = `Your Blood Group: ${data.blood_group}`;
                resultElement.style.display = 'block';
            } else {
                // Handle server errors
                throw new Error(data.error || 'Failed to detect blood group.');
            }
        } catch (error) {
            // Display error messages to the user
            errorElement.textContent = `Error: ${error.message}`;
        }
    });
});
