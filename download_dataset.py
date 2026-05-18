import kagglehub
import shutil
import os

print("Downloading FER-2013 dataset...")
path = kagglehub.dataset_download("msambare/fer2013")

print("Path to downloaded files:", path)

target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
if not os.path.exists(target_dir):
    print(f"Copying dataset to {target_dir}...")
    shutil.copytree(path, target_dir)
    print("Dataset copied successfully!")
else:
    print(f"Dataset directory {target_dir} already exists. You are ready to train.")
