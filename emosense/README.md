# EmoSense - Facial Emotion Detection

A web app that detects emotions (angry, disgust, fear, happy, neutral, sad, surprise) from your webcam in real time.
Built with a CNN (TensorFlow/Keras), OpenCV face detection and a Flask web interface.

## Features
- Register / log in
- Real-time webcam emotion detection with confidence bars
- History of saved detections
- Analytics: emotion breakdown and average confidence
- Settings: change your name, turn history saving on or off

## Project structure
```
emosense/
  app.py            Flask app (pages, login, /predict API)
  train.py          trains the CNN on FER-2013
  requirements.txt
  templates/        HTML pages
  static/           style.css, detect.js
```

## Setup
1. Install Python 3.10-3.12 and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        (Windows)
   source venv/bin/activate     (macOS / Linux)
   ```
2. Install the libraries: `pip install -r requirements.txt`
3. Download the FER-2013 image dataset from Kaggle and unzip it so the folders look like:
   `fer2013/train/<emotion>/*.png` and `fer2013/test/<emotion>/*.png`
4. Train the model: `python train.py`
   This creates `emotion_model.keras` and `labels.txt`. Training on CPU is slow; Google Colab's free GPU is faster
   (then download both files into this folder).
5. Start the app: `python app.py` and open http://127.0.0.1:5000

## Notes
- Allow camera access when the browser asks. The camera works on `localhost` without HTTPS.
- FER-2013 is noisy; expect roughly 60-70% test accuracy from this CNN.
- Set a real secret key before deploying: `SECRET_KEY=your-random-string`.

## Dataset
Kaggle: Facial Expression Recognition Challenge (FER-2013).
