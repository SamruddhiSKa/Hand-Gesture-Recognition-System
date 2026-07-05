# Hand Gesture Recognition System

A real-time hand gesture recognition app that converts hand signs into text using MediaPipe hand tracking and a machine learning classifier. The app runs in a Streamlit web interface and supports live webcam input, word building, and sentence history.

## Features

- Live webcam-based hand detection
- Real-time gesture prediction from hand landmarks
- Hold-to-add gesture capture for letter entry
- Word builder and completed sentence history
- Front and back camera toggle

## Project Structure

- app.py - Main Streamlit application
- core/ - Core processing and model loading logic
  - hand_processor.py - MediaPipe hand detection and landmark processing
  - model_manager.py - Model loading utilities
- config/ - App configuration and paths
- dataset/ - Collected gesture image/landmark data
- tools/ - Data collection, landmark extraction, and model training scripts
- models/ - Trained model file (generated after training)

## Requirements

Python 3.9+ is recommended.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

If the trained model file is missing, train a new model using the provided script:

```bash
python tools/train_alphabet_model.py
```

This will:
- load the landmark dataset from landmark_dataset.csv
- train a RandomForest classifier
- save the model to models/gesture_model.pkl

## Running the App

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then:
1. Allow camera access in the browser
2. Select the preferred camera
3. Hold a recognized gesture to capture letters automatically
4. Use the buttons to add letters, end a word, or clear everything

## Data Collection Tools

The repository includes helper scripts for building the dataset:

- tools/collect_alphabets_data.py - Collect gesture images for alphabet classes
- tools/extract_alphabets_landmarks.py - Extract hand landmarks from collected data
- tools/train_alphabet_model.py - Train the classifier

## Notes

- The app expects the trained model at models/gesture_model.pkl.
- If the model is not present, the app will show a warning until you train or place a model file in the expected location.

