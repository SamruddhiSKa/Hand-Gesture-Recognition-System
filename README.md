# Hand Gesture Recognition System

A real-time hand gesture recognition app that converts hand signs into text using MediaPipe hand tracking and a machine learning classifier. The app runs in a Streamlit web interface and supports live webcam input, word building, and sentence history.

## 🚀 Live Demo
[Open Live Demo](https://chatgpt.com/YOUR_STREAMLIT_URL_HERE)

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

Python 3.12 is the deployment target because MediaPipe 0.10.18 provides a Linux CPython 3.12 wheel.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

For local training or benchmarking, install the development dependencies and train a new model if needed:

```bash
pip install -r requirements-dev.txt
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

## Deploying to Streamlit Community Cloud

1. Push this repository to a GitHub repository. Make sure `models/gesture_model.pkl` is committed; it is required at runtime.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Select **New app**, choose the repository and branch, and set `app.py` as the main file path.
4. Deploy the app. Streamlit Community Cloud installs the runtime packages from `requirements.txt` and Linux packages from `packages.txt`.
5. Open the generated `streamlit.app` URL and allow camera access when the browser asks. The app needs HTTPS and browser camera permission for webcam input.

The pinned runtime dependencies target Python 3.12. MediaPipe 0.10.18 is a compatible 0.10.x release that preserves the existing legacy Hands Solutions API. `requirements-dev.txt` adds the pandas dependency needed by the offline training and benchmark tools. Streamlit Community Cloud requires a reachable ICE path for WebRTC; the app uses public STUN servers and can append a TURN server securely from Streamlit secrets when STUN-only negotiation is blocked. MediaPipe and webcam performance also depends on the viewer's browser and device. The benchmark measures only precomputed-landmark classifier inference, not deployed end-to-end performance.

### WebRTC Cloud configuration

If the deployed app stays on `Connecting to camera...` and Cloud logs show aioice transport errors, add these secrets in the app's Streamlit Cloud settings:

```toml
TURN_SERVER_URL = "turn:your-provider.example:3478"
TURN_SERVER_USERNAME = "your-username"
TURN_SERVER_CREDENTIAL = "your-credential"
```

Use credentials from a TURN provider; do not commit them to the repository. The app continues to use STUN when these secrets are absent.

## Benchmarking

Run the offline benchmark against the repository dataset and trained model:

```bash
python tools/benchmark.py --repeats 5
```

The command writes `benchmark_results.json` and prints a report containing held-out classification quality, warm model-inference latency percentiles, throughput, a majority-class baseline, dataset distribution, and model size. It measures precomputed landmark vectors only; it does not measure webcam capture, MediaPipe hand detection, WebRTC/browser overhead, network latency, or concurrent-user capacity. See `BENCHMARK_REPORT.md` for the recorded run and limitations.

## Data Collection Tools

The repository includes helper scripts for building the dataset:

- tools/collect_alphabets_data.py - Collect gesture images for alphabet classes
- tools/extract_alphabets_landmarks.py - Extract hand landmarks from collected data
- tools/train_alphabet_model.py - Train the classifier

## Notes

- The app expects the trained model at models/gesture_model.pkl.
- If the model is not present, the app will show a warning until you train or place a model file in the expected location.

