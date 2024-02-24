import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import matplotlib.pyplot as plt

label_mapping = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
        'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
        'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
        'nothing': 27, 'space': 28
    }


# Load your trained model
model = load_model('sign_language.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


# Reverse the label mapping to map from integer labels back to sign language symbols
reverse_label_mapping = {v: k for k, v in label_mapping.items()}


# Function to preprocess image and make predictions
def predict_sign_language(image):
     # Display the processed hand image before prediction
    # plt.imshow(image, cmap='gray')
    # plt.title("Processed Hand Image")
    # plt.show()
    
    # Resize image (make sure the size matches your model's input size)
    image = cv2.resize(image, (200, 200))
    
    # Expand dimensions to add channel information (1, 200, 200, 1) for grayscale
    image = np.expand_dims(image, axis=-1)
    
    # Normalize image
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image)
    
    # Process prediction (modify according to your model's output)
    predicted_class = np.argmax(prediction)
    
    # Map the predicted class integer back to a label
    predicted_label = reverse_label_mapping[predicted_class]
    return predicted_label

# Example usage
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform hand detection
    results = hands.process(frame_rgb)

    # Check if a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the bounding box of the hand
            h, w, _ = frame.shape
            minX = minY = float('inf')
            maxX = maxY = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < minX:
                    minX = x
                if y < minY:
                    minY = y
                if x > maxX:
                    maxX = x
                if y > maxY:
                    maxY = y

            # Add some padding to the bounding box
            padding = 20
            minX = max(0, minX - padding)
            minY = max(0, minY - padding)
            maxX = min(w, maxX + padding)
            maxY = min(h, maxY + padding)

            # Extract the hand region
            hand_image = frame[minY:maxY, minX:maxX]

             # Convert hand region to grayscale
            hand_image_gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)

            # Make prediction on the hand image
            prediction = predict_sign_language(hand_image_gray)

            # Display the bounding box and prediction on the original frame
            cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 255, 0), 2)
            cv2.putText(frame, f"Predicted: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    cv2.imshow('Sign Language Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
