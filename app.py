from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp

app = Flask(__name__)

# Load trained model and label encoder
model = tf.keras.models.load_model('gesture_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Define label map for display
label_map = {
    'palm': 'PALM',
    'love': 'LOVE',
    'swag': 'SWAG',
    'call_me': 'CALL ME',
    'thumbs_down': 'THUMBS DOWN',
    'okay': 'OKAY',
    'peace': 'PEACE',
    'fist': 'FIST',
    'thumbs_up': 'thumbs_up'
}

# Start video capture
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x = [lm.x for lm in hand_landmarks.landmark]
                y = [lm.y for lm in hand_landmarks.landmark]
                features = np.array(x + y).reshape(1, -1)

                prediction = model.predict(features)[0]
                predicted_class = np.argmax(prediction)
                confidence = prediction[predicted_class] * 100
                label = label_encoder.inverse_transform(
                    [predicted_class])[0].strip().lower()

                display_text = f"{label_map.get(label, label.upper())} ({confidence:.2f}%)"

                cv2.putText(frame, display_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame to JPEG and yield for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
