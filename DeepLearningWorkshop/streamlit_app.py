import streamlit as st
import os
import json
import numpy as np
import uuid
import shutil
from PIL import Image
from model_wrapper import SignatureModel

st.set_page_config(page_title="Signature Auth", layout="centered", page_icon="✍️")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, 'users.json')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model (cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    return SignatureModel(MODEL_PATH)

try:
    sig_model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

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
    
    embedding = np.array(embedding)
    
    for username, data in db.items():
        stored_embedding = np.array(data['embedding'])
        dist = np.linalg.norm(embedding - stored_embedding)
        
        if dist < min_dist:
            min_dist = dist
            best_match = username
            
    if min_dist > threshold:
        return None, min_dist
        
    return best_match, min_dist

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

# Routing
if st.session_state['logged_in']:
    st.title("Success! 🎉")
    st.markdown(f"### Welcome back, {st.session_state['username']}!")
    st.write("The system analyzed your signature's unique features, generated a 128-dimensional embedding, and found the closest match in the secure database using Euclidean distance.")
    
    if st.button("Log Out", type="primary"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()

else:
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Login", "Register"])
    
    if page == "Login":
        st.title("Signature Login")
        st.write("Upload your signature to authenticate.")
        
        uploaded_file = st.file_uploader("Choose a signature image...", type=["jpg", "jpeg", "png"], key="login_upload")
        
        # Test Galleries
        with st.expander("Help for Testers? 💡"):
            st.write("**Genuine Signatures**")
            reg_dir = os.path.join(BASE_DIR, 'static', 'signatures', 'registered')
            reg_files = [f for f in os.listdir(reg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(reg_dir) else []
            
            cols = st.columns(5)
            selected_test_sig = None
            selected_test_path = None
            
            for i, f in enumerate(reg_files[:10]):
                with cols[i % 5]:
                    st.image(os.path.join(reg_dir, f), use_container_width=True)
                    if st.button("Select", key=f"btn_reg_{f}"):
                        selected_test_sig = f
                        selected_test_path = os.path.join(reg_dir, f)
            
            st.write("---")
            st.markdown("**Test Robustness with Forged Signatures**")
            forged_dir = os.path.join(BASE_DIR, 'static', 'signatures', 'forged_registered')
            forged_files = [f for f in os.listdir(forged_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(forged_dir) else []
            
            cols_f = st.columns(5)
            for i, f in enumerate(forged_files[:10]):
                with cols_f[i % 5]:
                    st.image(os.path.join(forged_dir, f), use_container_width=True)
                    if st.button("Select", key=f"btn_forg_{f}"):
                        selected_test_sig = f
                        selected_test_path = os.path.join(forged_dir, f)

        # Login Logic
        if st.button("Identify & Login", type="primary") or selected_test_path:
            sig_path = None
            is_uploaded = False
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
                sig_path = os.path.join(UPLOAD_FOLDER, temp_filename)
                with open(sig_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                is_uploaded = True
            elif selected_test_path:
                sig_path = selected_test_path
            else:
                st.error("Please upload a signature or select one from the tester gallery.")
            
            if sig_path:
                with st.spinner("Authenticating..."):
                    embedding = sig_model.get_embedding(sig_path)
                    
                    if is_uploaded and os.path.exists(sig_path):
                        os.remove(sig_path)
                        
                    db = load_db()
                    if not db:
                        st.error("Database is empty. Please register first.")
                    else:
                        username, distance = get_closest_match(embedding, db, threshold=15.0)
                        
                        if username:
                            st.success(f"Match found! Distance: {distance:.2f}")
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            st.rerun()
                        else:
                            st.error("No matching account found. Run away! 🚨")

    elif page == "Register":
        st.title("Create Account")
        
        username_input = st.text_input("Username", placeholder="Enter your name")
        
        st.write("Upload Signature Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="reg_upload")
        
        with st.expander("Need a test signature? 💡"):
            st.write("For testing purposes, use these fresh signatures:")
            new_dir = os.path.join(BASE_DIR, 'static', 'signatures', 'new_users')
            new_files = [f for f in os.listdir(new_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(new_dir) else []
            
            cols = st.columns(5)
            selected_test_sig = None
            selected_test_path = None
            
            for i, f in enumerate(new_files[:10]):
                with cols[i % 5]:
                    st.image(os.path.join(new_dir, f), use_container_width=True)
                    if st.button("Select", key=f"btn_new_{f}"):
                        selected_test_sig = f
                        selected_test_path = os.path.join(new_dir, f)
        
        if st.button("Register Account", type="primary") or selected_test_path:
            if not username_input:
                st.error("Please enter a username.")
            else:
                sig_path = None
                is_uploaded = False
                original_filename = ""
                
                if uploaded_file is not None:
                    original_filename = uploaded_file.name
                    temp_filename = f"{uuid.uuid4()}_{original_filename}"
                    sig_path = os.path.join(UPLOAD_FOLDER, temp_filename)
                    with open(sig_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    is_uploaded = True
                elif selected_test_path:
                    sig_path = selected_test_path
                    original_filename = selected_test_sig
                else:
                    st.error("Please upload a signature or select one from the tester gallery.")
                
                if sig_path:
                    db = load_db()
                    if username_input in db:
                        st.error("Username already exists.")
                    else:
                        with st.spinner("Processing signature..."):
                            embedding = sig_model.get_embedding(sig_path)
                            
                            final_filename = original_filename
                            if is_uploaded:
                                registered_dir = os.path.join(BASE_DIR, 'static', 'signatures', 'registered')
                                if not os.path.exists(registered_dir):
                                    os.makedirs(registered_dir)
                                new_sig_path = os.path.join(registered_dir, final_filename)
                                # Make sure not to overwrite if file exists with same name
                                counter = 1
                                while os.path.exists(new_sig_path):
                                    name, ext = os.path.splitext(original_filename)
                                    final_filename = f"{name}_{counter}{ext}"
                                    new_sig_path = os.path.join(registered_dir, final_filename)
                                    counter += 1
                                    
                                shutil.move(sig_path, new_sig_path)
                            
                            db[username_input] = {
                                'embedding': embedding.tolist(),
                                'signature_file': final_filename
                            }
                            save_db(db)
                            
                            st.success(f"User {username_input} registered successfully!")
                            st.info("You can now go to the Login page to authenticate.")
