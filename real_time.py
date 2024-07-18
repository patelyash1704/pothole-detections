import tensorflow as tf
import numpy as np
import cv2 as cv

model = tf.keras.models.load_model('pothole_model_4.h5')


def preprocess_image(image):
    resized_image = tf.image.resize(image, (256, 256))
    return resized_image / 255.0


def classify_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))[0][0]
    return prediction


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    prediction = classify_image(frame)
    accuracy = 1 - np.abs(prediction - 0.5) * 2

    if prediction > 0.5:
        cv.putText(frame, f"Pothole ({accuracy:.2f})", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv.putText(frame, f"Normal ({accuracy:.2f})", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Classification Result', frame)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
