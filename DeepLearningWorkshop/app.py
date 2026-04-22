from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import json
import numpy as np
from model_wrapper import SignatureModel

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-signature-auth'

# Initialize model
MODEL_PATH = 'best_model.keras'
sig_model = SignatureModel(MODEL_PATH)

DB_PATH = 'users.json'

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_PATH, 'w') as f:
        json.dump(db, f)

def get_closest_match(embedding, db, threshold=15.0):
    best_match = None
    min_dist = float('inf')
    
    # Convert input embedding to numpy array
    embedding = np.array(embedding)
    
    for username, data in db.items():
        stored_embedding = np.array(data['embedding'])
        # Euclidean distance
        dist = np.linalg.norm(embedding - stored_embedding)
        
        if dist < min_dist:
            min_dist = dist
            best_match = username
            
    # Enforce strict matching threshold
    if min_dist > threshold:
        return None, min_dist
        
    return best_match, min_dist

@app.route('/')
def index():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login')
def login():
    # List images in static/signatures/registered
    sig_dir = 'static/signatures/registered'
    if not os.path.exists(sig_dir):
        os.makedirs(sig_dir)
    signatures = [f for f in os.listdir(sig_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # List images in static/signatures/forged_registered
    forged_dir = 'static/signatures/forged_registered'
    if not os.path.exists(forged_dir):
        os.makedirs(forged_dir)
    forged_signatures = [f for f in os.listdir(forged_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    return render_template('login.html', signatures=signatures, forged_signatures=forged_signatures)

@app.route('/register')
def register():
    # List images in static/signatures/new_users
    sig_dir = 'static/signatures/new_users'
    if not os.path.exists(sig_dir):
        os.makedirs(sig_dir)
    signatures = [f for f in os.listdir(sig_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('register.html', signatures=signatures)

import uuid
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'signature' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['signature']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'filepath': filepath, 'filename': filename})

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    username = data.get('username')
    sig_filename = data.get('signature')
    is_uploaded = data.get('is_uploaded', False)
    
    if not username or not sig_filename:
        return jsonify({'error': 'Missing username or signature'}), 400
    
    db = load_db()
    if username in db:
        return jsonify({'error': 'Username already exists'}), 400
    
    if is_uploaded:
        sig_path = os.path.join(app.config['UPLOAD_FOLDER'], sig_filename)
    else:
        sig_path = os.path.join('static/signatures/new_users', sig_filename)
    
    embedding = sig_model.get_embedding(sig_path)
    
    # If the signature was uploaded, move it to the registered folder permanently
    if is_uploaded:
        import shutil
        registered_dir = 'static/signatures/registered'
        if not os.path.exists(registered_dir):
            os.makedirs(registered_dir)
        new_sig_path = os.path.join(registered_dir, sig_filename)
        shutil.move(sig_path, new_sig_path)
    
    db[username] = {
        'embedding': embedding.tolist(),
        'signature_file': sig_filename
    }
    save_db(db)
    
    return jsonify({'success': True, 'message': f'User {username} registered successfully!'})

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    sig_filename = data.get('signature')
    is_uploaded = data.get('is_uploaded', False)
    is_forged = data.get('is_forged', False)
    
    if not sig_filename:
        return jsonify({'error': 'No signature provided'}), 400
    
    if is_uploaded:
        sig_path = os.path.join(app.config['UPLOAD_FOLDER'], sig_filename)
    elif is_forged:
        sig_path = os.path.join('static/signatures/forged_registered', sig_filename)
    else:
        sig_path = os.path.join('static/signatures/registered', sig_filename)
    
    embedding = sig_model.get_embedding(sig_path)
    
    # Uploads for login are strictly temporary, delete immediately after embedding
    if is_uploaded and os.path.exists(sig_path):
        os.remove(sig_path)
    
    db = load_db()
    if not db:
        return jsonify({'error': 'Database is empty. Please register first.'}), 400
    
    # We use a strict threshold here (e.g., 15.0) to prevent random matching
    username, distance = get_closest_match(embedding, db, threshold=15.0)
    
    if username:
        session['username'] = username
        return jsonify({'success': True, 'username': username, 'distance': float(distance)})
    
    return jsonify({'error': 'No matching account found. Run away!'}), 401

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
