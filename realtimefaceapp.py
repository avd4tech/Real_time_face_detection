import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np

# Load the model architecture and weights
json_file_path = "emotiondetector.json"
with open(json_file_path, "r", encoding="utf-8") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the input image for the model
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Streamlit app layout
st.title("Real-time Facial Emotion Detector")
run = st.button('Start Detection')
stop = st.button('Stop Detection')

if run:
    # Initialize webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        st.write("Error: Could not open webcam.")
    else:
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        frame_window = st.image([])

        while run and not stop:
            ret, im = webcam.read()
            if not ret:
                st.write("Error: Could not read frame.")
                break

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            
            frame_window.image(im, channels='BGR')
        
        webcam.release()
        cv2.destroyAllWindows()

elif stop:
    st.write("Detection stopped.")
