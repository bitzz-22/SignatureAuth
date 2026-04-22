document.addEventListener('DOMContentLoaded', () => {
    const sigItems = document.querySelectorAll('.sig-item');
    const helpBtn = document.getElementById('help-btn');
    const testerSection = document.getElementById('tester-section');
    const dropZone = document.getElementById('drop-zone');
    const sigUpload = document.getElementById('sig-upload');
    const previewImg = document.getElementById('selected-preview');
    const submitBtn = document.getElementById('submit-btn');
    
    let selectedSignature = null;
    let isUploaded = false;
    let isForged = false;

    // Toggle Help Section
    if (helpBtn) {
        helpBtn.addEventListener('click', () => {
            const isHidden = testerSection.style.display === 'none';
            testerSection.style.display = isHidden ? 'block' : 'none';
        });
    }

    // Gallery Selection
    sigItems.forEach(item => {
        item.addEventListener('click', () => {
            sigItems.forEach(i => i.classList.remove('selected'));
            item.classList.add('selected');
            selectedSignature = item.dataset.filename;
            isUploaded = false;
            isForged = item.dataset.source === 'forged';
            
            // Show preview of gallery item
            previewImg.src = item.querySelector('img').src;
            previewImg.style.display = 'inline-block';
            submitBtn.disabled = false;
        });
    });

    // Upload Logic
    if (dropZone) {
        dropZone.addEventListener('click', () => sigUpload.click());

        sigUpload.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                await handleFileUpload(file);
            }
        });

        // Drag and Drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragging');
        });

        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));

        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragging');
            const file = e.dataTransfer.files[0];
            if (file) {
                await handleFileUpload(file);
            }
        });
    }

    async function handleFileUpload(file) {
        const formData = new FormData();
        formData.append('signature', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (result.success) {
                selectedSignature = result.filename;
                isUploaded = true;
                
                // Show Preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImg.src = e.target.result;
                    previewImg.style.display = 'inline-block';
                };
                reader.readAsDataURL(file);
                
                // Clear gallery selection
                sigItems.forEach(i => i.classList.remove('selected'));
                submitBtn.disabled = false;
            }
        } catch (err) {
            console.error('Upload failed', err);
        }
    }

    // Form Submissions
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const messageEl = document.getElementById('message');
            messageEl.textContent = 'Authenticating...';
            messageEl.className = '';

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        signature: selectedSignature,
                        is_uploaded: isUploaded,
                        is_forged: isForged
                    })
                });

                const result = await response.json();
                if (result.success) {
                    messageEl.textContent = `Welcome back, ${result.username}!`;
                    messageEl.className = 'success';
                    setTimeout(() => window.location.href = '/', 1000);
                } else {
                    messageEl.textContent = result.error || 'Login failed';
                    messageEl.className = 'error';
                }
            } catch (err) {
                messageEl.textContent = 'Server error';
                messageEl.className = 'error';
            }
        });
    }

    const registerForm = document.getElementById('register-form');
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const messageEl = document.getElementById('message');

            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        username: username,
                        signature: selectedSignature,
                        is_uploaded: isUploaded
                    })
                });

                const result = await response.json();
                if (result.success) {
                    messageEl.textContent = 'Registration successful!';
                    messageEl.className = 'success';
                    setTimeout(() => window.location.href = '/login', 1500);
                } else {
                    messageEl.textContent = result.error || 'Failed';
                    messageEl.className = 'error';
                }
            } catch (err) {
                messageEl.textContent = 'Server error';
                messageEl.className = 'error';
            }
        });
    }
});
