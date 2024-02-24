import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

# Load your trained model
model = load_model('sign_language.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Reverse the label mapping
label_mapping = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
    'nothing': 27, 'space': 28
}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Function to preprocess image and make predictions
def predict_sign_language(image):
    # The image is already resized to (227, 227) before this function is called
    
    # Normalize image
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image)
    
    # Process prediction
    predicted_class = np.argmax(prediction)
    predicted_label = reverse_label_mapping[predicted_class]
    
    return predicted_label

cap = cv2.VideoCapture(0)

# Frame rate control variables
frame_rate = 10
prev_frame_time = 0

while True:
    time_elapsed = time.time() - prev_frame_time
    ret, frame = cap.read()
    if not ret or time_elapsed < 1./frame_rate:
        continue
    prev_frame_time = time.time()

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            minX, minY = w, h
            maxX = maxY = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                minX, minY = min(minX, x), min(minY, y)
                maxX, maxY = max(maxX, x), max(maxY, y)

            # Determine the size for a square bounding box
            squareSide = 270  # Fixed size before resizing
            
            # Center the square box around the hand's center
            centerX = (minX + maxX) // 2
            centerY = (minY + maxY) // 2

            # Calculate the square bounding box coordinates
            squareMinX = max(0, centerX - squareSide // 2)
            squareMinY = max(0, centerY - squareSide // 2)
            squareMaxX = squareMinX + squareSide
            squareMaxY = squareMinY + squareSide

            # Adjust if the square goes beyond the frame dimensions
            if squareMaxX > w:
                squareMaxX = w
                squareMinX = w - squareSide
            if squareMaxY > h:
                squareMaxY = h
                squareMinY = h - squareSide

            hand_image = frame[squareMinY:squareMaxY, squareMinX:squareMaxX]
            
            # Resize the square hand image to (227, 227) for the model
            hand_image_resized = cv2.resize(hand_image, (227, 227))
            
            # Optionally, display the resized hand image before prediction
            cv2.imshow('Resized Hand Image Before Prediction', hand_image_resized)

            prediction = predict_sign_language(hand_image_resized)

            cv2.rectangle(frame, (squareMinX, squareMinY), (squareMaxX, squareMaxY), (0, 255, 0), 2)
            cv2.putText(frame, f"Predicted: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
