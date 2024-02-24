import cv2
import numpy as np
from keras.models import load_model

# Load your trained model
model = load_model('model.h5')

# Function to preprocess image and make predictions
def predict_sign_language(image):
    # Preprocess input image (modify according to your model's requirements)
    image = cv2.resize(image, (200, 200))  # Example resizing
    image = image.astype('float32') / 255.0  # Example normalization

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)
    # Process prediction (modify according to your model's output)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Example usage
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Make prediction on each frame
    prediction = predict_sign_language(frame)

    # Display prediction (modify according to your model's output)
    cv2.putText(frame, f"Predicted class: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
