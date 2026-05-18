import cv2
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

# Load the trained model
MODEL_PATH = 'emotion_model.h5'
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please run train.py first or download a pre-trained model.")
    exit()

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Define emotion classes mapping
# Note: Ensure this matches the alphabetical order of your training directories
emotion_classes = ['Angry', 'Happy', 'Neutral', 'Sad']

# Load OpenCV's pre-trained Haar Cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(cascade_path)

if face_classifier.empty():
    print("Error loading haarcascade face detection model.")
    exit()

# Start video capture from webcam
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam. Exiting...")
        break
        
    # Convert frame to grayscale for face detection and model prediction
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the region of interest (ROI) - the face
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        try:
            # Resize ROI to match model's expected input shape (48x48)
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        except Exception as e:
            continue
            
        # Preprocess ROI for prediction
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = tf.keras.preprocessing.image.img_to_array(roi)
            roi = np.expand_dims(roi, axis=0) # Add batch dimension (1, 48, 48, 1)
            
            # Predict emotion
            prediction = model.predict(roi, verbose=0)[0]
            max_index = np.argmax(prediction)
            predicted_emotion = emotion_classes[max_index]
            confidence = prediction[max_index] * 100
            
            # Set color based on emotion
            color = (0, 255, 0) # Green default
            if predicted_emotion == 'Angry': color = (0, 0, 255) # Red
            elif predicted_emotion == 'Sad': color = (255, 0, 0) # Blue
            elif predicted_emotion == 'Happy': color = (0, 255, 255) # Yellow
            elif predicted_emotion == 'Neutral': color = (255, 255, 255) # White
            
            label = f"{predicted_emotion} ({confidence:.1f}%)"
            
            # Display text on screen above the bounding box
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, 'No Face Detected', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
    # Show the video feed
    cv2.imshow('Face Emotion Detection System', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
