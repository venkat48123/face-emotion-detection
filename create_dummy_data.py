import os
import numpy as np
import cv2

def create_synthetic_dataset(base_dir='dataset', num_train=50, num_test=10):
    emotions = ['angry', 'happy', 'neutral', 'sad']
    splits = {'train': num_train, 'test': num_test}
    
    for split, count in splits.items():
        for emotion in emotions:
            dir_path = os.path.join(base_dir, split, emotion)
            os.makedirs(dir_path, exist_ok=True)
            
            for i in range(count):
                # Create a random noisy 48x48 image
                # In a real scenario, these would be actual face images
                img = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
                
                # Add some basic shapes to differentiate classes slightly (just so the model learns *something*)
                if emotion == 'happy':
                    cv2.circle(img, (24, 24), 10, 255, -1)
                elif emotion == 'sad':
                    cv2.rectangle(img, (14, 14), (34, 34), 0, -1)
                elif emotion == 'angry':
                    cv2.line(img, (0, 0), (48, 48), 128, 5)
                elif emotion == 'neutral':
                    pass # Just noise
                
                file_path = os.path.join(dir_path, f"{emotion}_{i}.jpg")
                cv2.imwrite(file_path, img)
                
    print(f"Synthetic dataset created at '{base_dir}/'")
    print(f"Total train images per class: {num_train}")
    print(f"Total test images per class: {num_test}")

if __name__ == "__main__":
    create_synthetic_dataset()
