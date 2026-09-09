import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path

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
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.append([lm.x, lm.y])  # # Extracting X and Y for all 21 points
                
                coords = np.array(coords)
                coords = coords - coords[0]
                scale = np.max(np.abs(coords))
                if scale != 0: coords = coords / scale
                
                data.append(coords.flatten())
                labels.append(label)

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv(OUTPUT_PATH, index=False)
print("Dataset created:", len(df))