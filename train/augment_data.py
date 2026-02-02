import os
import numpy as np

def augment_my_data(base_path, factor=10):
    actions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for action in actions:
        action_path = os.path.join(base_path, action)
        sequences = [s for s in os.listdir(action_path) if s.isdigit()]
        
        if not sequences: continue
        print(f"Augmenting {action}...")
        
        # Move the index counter OUTSIDE the sequence loop
        next_idx = max([int(s) for s in sequences]) + 1
        
        for seq in sequences:
            seq_path = os.path.join(action_path, seq)
            original_frames = []
            
            # Load frames
            for i in range(30):
                file_path = os.path.join(seq_path, f"{i}.npy")
                if os.path.exists(file_path):
                    original_frames.append(np.load(file_path))
            
            if len(original_frames) < 30: continue

            # Create factor versions for THIS sequence
            for _ in range(factor):
                new_seq_path = os.path.join(action_path, str(next_idx))
                os.makedirs(new_seq_path, exist_ok=True)
                
                for i, frame in enumerate(original_frames):
                    noise = np.random.normal(0, 0.002, frame.shape)
                    np.save(os.path.join(new_seq_path, f"{i}.npy"), frame + noise)
                
                next_idx += 1 # Increment globally for the action
                
    print("Done! Data expansion complete.")

if __name__ == "__main__":
    # Ensure this points to where your .npy files actually live
    augment_my_data("SignData")