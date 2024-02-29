import cv2
from matplotlib import pyplot as plt
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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

# Initialize Tkinter GUI
root = tk.Tk()
root.title("Sign Language Prediction")

# Create a main frame with two columns
main_frame = ttk.Frame(root)
main_frame.pack(pady=20)

# Frame for displaying the video feed and prediction
video_frame = ttk.Label(main_frame)
video_frame.grid(row=0, column=0, columnspan=2)

# Update for displaying the sentence
sentence = ''  # Initialize sentence variable

sentence_frame = ttk.Frame(main_frame)
sentence_frame.grid(row=1, column=0, sticky="ew")
sentence_label = tk.Label(sentence_frame, text='', font=('Helvetica', 24))
sentence_label.pack()

# Initialize maximum sentence length and create a scale to adjust it
max_sentence_length = tk.IntVar(value=20 )  # Starting value

length_frame = ttk.Frame(main_frame)
length_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
length_label = tk.Label(length_frame, font=('Helvetica', 14))
length_label.pack(side=tk.LEFT, padx=(0, 10))


# Function to preprocess image and make predictions
def predict_sign_language(image):
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = reverse_label_mapping[predicted_class]
    return predicted_label

first_prediction_time = None
first_prediction = None
last_prediction = None

def update_sentence(prediction):
    global sentence, first_prediction_time, first_prediction, last_prediction
    current_time = time.time()
    
    if first_prediction_time is None or (current_time - first_prediction_time) <= 1:
        if first_prediction_time is None:
            first_prediction_time = current_time
            first_prediction = prediction
            return
        last_prediction = prediction
        # After exactly 10 seconds, compare first and last prediction if needed
        if (current_time - first_prediction_time) >= 0.5:
            if prediction != "nothing":
                if first_prediction == last_prediction:
                    # Avoid adding consecutive duplicates
                    if len(sentence) > 0:
                        last_char = sentence[-1]
                        if (prediction == "space" and last_char == " ") or (sentence.endswith(prediction)):
                            return  # Skip adding the same character or space consecutively
                    if prediction == "space":
                        sentence += ' '
                    else:
                        sentence += prediction

                    sentence = sentence[-max_sentence_length.get():]
                    sentence_label.config(text=sentence)

            first_prediction_time = current_time
            first_prediction = prediction
            last_prediction = None
    
    elif (current_time - first_prediction_time) > 1:
        first_prediction_time = current_time
        first_prediction = prediction
        last_prediction = None
    return

def reset_sentence():
    global sentence
    sentence = ''
    sentence_label.config(text=sentence)

def delete_last_word():
    global sentence
    sentence = sentence[:-1]
    sentence_label.config(text=sentence)

reset_button = tk.Button(sentence_frame, text="Reset Sentence", command=reset_sentence)
reset_button.pack(side=tk.LEFT)

delete_word_button = tk.Button(sentence_frame, text="Delete Last Word", command=delete_last_word)
delete_word_button.pack(side=tk.RIGHT)

# Function to update the GUI with the OpenCV frame
def update_video_frame(frame):
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.imgtk = imgtk  # Keep a reference, prevent garbage-collection
    video_frame.configure(image=imgtk)
    root.update()

cap = cv2.VideoCapture(0)
frame_rate = 10
prev_frame_time = 0
prediction = ''

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
            squareSide = 300  # Fixed size before resizing
            
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

            hand_image = frame_rgb[squareMinY:squareMaxY, squareMinX:squareMaxX]
            hand_image_resized = cv2.resize(hand_image, (227, 227))  # Assuming `hand_image` is defined
            prediction = predict_sign_language(hand_image_resized)
            update_sentence(prediction)  # Update the GUI with the new prediction

    # Update the video frame in the GUI
    update_video_frame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()  # Close the GUI when done