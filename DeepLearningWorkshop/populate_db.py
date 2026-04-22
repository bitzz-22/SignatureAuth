import os
import json
import numpy as np
from model_wrapper import SignatureModel

def populate():
    MODEL_PATH = 'best_model.keras'
    sig_model = SignatureModel(MODEL_PATH)
    
    db = {}
    sig_dir = 'static/signatures/registered'
    usernames = ['Alex_Morgan', 'Jordan_Lee', 'Sam_Rivers', 'Casey_Wright', 'Taylor_Swift']
    
    files = [f for f in os.listdir(sig_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, f in enumerate(files):
        name = usernames[i] if i < len(usernames) else f'User_{i+1}'
        print(f"Registering {name} with {f}...")
        
        sig_path = os.path.join(sig_dir, f)
        embedding = sig_model.get_embedding(sig_path)
        
        db[name] = {
            'embedding': embedding.tolist(),
            'signature_file': f
        }
    
    with open('users.json', 'w') as f:
        json.dump(db, f, indent=4)
    
    print(f"Database populated successfully with {len(db)} users!")

if __name__ == "__main__":
    populate()
