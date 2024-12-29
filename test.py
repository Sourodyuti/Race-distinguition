import cv2
import numpy as np
from keras.models import model_from_json

# Define a dictionary for your skin tone labels
skin_tone_dict = {0: "Light", 1: "Medium", 2: "Dark"}  # Adjust keys and labels based on your model

# Load the trained model
# Load model structure from JSON
json_file = open('model/skin_type_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
skin_tone_model = model_from_json(loaded_model_json)

# Load model weights
skin_tone_model.load_weights("model/skin_type_model.h5")
print("Loaded skin tone model from disk")

# Start the webcam feed or load a video
# Uncomment for live webcam feed
# cap = cv2.VideoCapture(0)

# Uncomment to use a video file
cap = cv2.VideoCapture("C:\\Users\\souro\\Downloads\\test3.webm")

while True:
    # Read frames from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for display purposes
    frame = cv2.resize(frame, (1280, 720))

    # Initialize a Haar cascade for face detection
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        # Crop the face region and resize it to the model input size (48x48)
        roi_color_frame = frame[y:y + h, x:x + w]  # Use the RGB frame directly
        cropped_img = cv2.resize(roi_color_frame, (48, 48))

        # Preprocess the face for the model
        cropped_img = np.expand_dims(cropped_img, axis=0)  # Add batch dimension
        cropped_img = cropped_img / 255.0  # Normalize the image

        # Predict the skin tone
        skin_tone_prediction = skin_tone_model.predict(cropped_img)
        maxindex = int(np.argmax(skin_tone_prediction))

        # Display the skin tone on the frame
        cv2.putText(frame, skin_tone_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Skin Tone Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
