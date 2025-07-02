# ✋ Hand Gesture Recognition using Deep Learning and Flask

This project is a real-time hand gesture recognition system built using Deep Learning, MediaPipe, OpenCV, and Flask. It detects custom hand gestures through a webcam and displays predictions live in a stylish web interface.

---

## 📌 Features

- Real-time gesture detection using webcam
- Deep learning model trained on 9 custom gestures
- Flask-based web interface with a modern dark theme
- Responsive UI with live video stream and prediction overlay

---

## 🧠 Model Overview

- Framework: TensorFlow (Keras)
- Input: 42 features (x and y coordinates of 21 hand landmarks)
- Model Type: Dense Neural Network (DNN)
- Activation Functions: ReLU + Softmax
- Accuracy: ~95% on test data

### Supported Gestures (Labels)

- ✋ palm  
- ✊ fist  
- 👌 okay  
- ✌️ peace  
- 🖕 fuck  
- 🤘 swag  
- 👍 thumbs_up  
- 👎 thumbs_down  
- 🤙 call_me  

---

## 📁 Project Structure
hand-gesture-recognition/
├── app.py
├── gesture_model.h5
├── label_encoder.pkl
├── requirements.txt
├── README.md
├── gesture_data.csv # (optional)
├── templates/
│ └── index.html
├── static/
│ └── style.css

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/omahuja22/hand-gesture-recognition.git
cd hand-gesture-recognition

2. Install Dependencies
pip install -r requirements.txt

3. Run the App
python app.py

4. View in Browser
Open your browser and visit

results

🧰 Technologies Used
Python 3.10+
TensorFlow 2.x
OpenCV
MediaPipe
Flask (Web Framework)
NumPy, Joblib

💡 Future Enhancements
Multi-hand detection support
Gesture history and statistics
Voice feedback for accessibility
Cloud deployment with Render or ngrok
Mobile compatibility

📜 License
This project is licensed under the MIT License.
Feel free to use, share, or modify it for academic and non-commercial purposes.

🙋‍♂️ Author
Om Ahuja
GitHub: github.com/omahuja.22
Email: oma48446@gmail.com
