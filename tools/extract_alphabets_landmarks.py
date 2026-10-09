import cv2
import os
import pandas as pd
from pathlib import Path
from core.landmark_features import extract_landmarks

# Robust imports
try:
    from mediapipe.python.solutions import hands as mp_hands
except:
    from mediapipe.solutions import hands as mp_hands

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "alphabets"
OUTPUT_PATH = ROOT_DIR / "landmark_dataset.csv"

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

data, labels = [], []

for label_path in sorted(DATASET_PATH.iterdir()):
    label = label_path.name
    if not os.path.isdir(label_path): continue
    
    print("Processing:", label)
    for img_path in sorted(label_path.iterdir()):
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                features = extract_landmarks(hand_landmarks, image.shape)
                data.append(features[0])
                labels.append(label)

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv(OUTPUT_PATH, index=False)
print("Dataset created:", len(df))