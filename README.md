# Face Emotion Detection System

A real-time face emotion detection system using Python, Convolutional Neural Networks (CNN), OpenCV, and TensorFlow/Keras.

## Project Structure
- `requirements.txt`: Python dependencies.
- `model.py`: Defines the CNN architecture for emotion classification.
- `train.py`: Script to train the model using data augmentation and hyperparameter tuning (Early Stopping, Learning Rate Reduction).
- `detect.py`: Real-time inference script using webcam and OpenCV face detection.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Preparation**
   You need a dataset (like FER2013) with images organized in the following structure inside the project directory:
   ```
   dataset/
   ├── train/
   │   ├── angry/
   │   ├── happy/
   │   ├── neutral/
   │   └── sad/
   └── test/
       ├── angry/
       ├── happy/
       ├── neutral/
       └── sad/
   ```

3. **Train the Model**
   Once the dataset is in place, train the model to achieve high accuracy:
   ```bash
   python train.py
   ```
   This will generate a trained model file named `emotion_model.h5`.

4. **Run Real-Time Detection**
   Run the webcam detection script:
   ```bash
   python detect.py
   ```
   Press `q` to quit the webcam window.
