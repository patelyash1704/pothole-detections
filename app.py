# Flask app
from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the model
model = load_model("pothole.h5")

# Flag to control real-time detection
keep_detecting = True

# Function to preprocess the image
def preprocess_image(image):
    resized_image = tf.image.resize(image, (128, 128))
    return resized_image / 255.0

# Function to detect pothole in the frame
def detect_pothole(frame):
    preprocessed_image = preprocess_image(frame)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))[0][0]
    return prediction

# Function to generate video feed with real-time detection
def gen_frames():
    cap = cv2.VideoCapture(0)
    while keep_detecting:
        success, frame = cap.read()
        if not success:
            break
        else:
            prediction = detect_pothole(frame)
            accuracy = 1 - np.abs(prediction - 0.5) * 2
            if prediction > 0.5:
                cv2.putText(frame, f"Pothole ({accuracy:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Normal ({accuracy:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
UPLOAD_FOLDER = 'upload'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def greet():
    return render_template('greet.html')

@app.route('/greet', methods=['POST'])
def classify_choice():
    choice = request.form['choice']
    if choice == 'image':
        return redirect(url_for('image_classification'))
    elif choice == 'realtime':
        return redirect(url_for('realtime_classification'))

@app.route('/image_classification', methods=['GET', 'POST'])
def image_classification():
    result = None
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        print("File:", file)  # Debugging: Print the file object
        if file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            print("File saved to:", filepath)  # Debugging: Print the file path

            img = cv2.imread(filepath)
            resize = tf.image.resize(img, (128, 128))
            prediction = model.predict(np.expand_dims(resize / 255, 0))[0][0]
            result = 'Pothole' if prediction > 0.5 else 'Normal'

    return render_template('image_classification.html', result=result)


@app.route('/realtime_classification')
def realtime_classification():
    return render_template('realtime_classification.html')

@app.route('/start_detection/<detection_type>', methods=['POST'])
def start_detection(detection_type):
    global keep_detecting
    keep_detecting = True
    if detection_type == 'realtime':
        return redirect(url_for('realtime_classification'))
    elif detection_type == 'image':
        return redirect(url_for('image_classification'))

@app.route('/stop_detection')
def stop_detection():
    global keep_detecting
    keep_detecting = False
    return redirect(url_for('realtime_classification'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
