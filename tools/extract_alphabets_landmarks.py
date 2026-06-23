import cv2
import os
import pandas as pd
import numpy as np

# Robust imports
try:
    from mediapipe.python.solutions import hands as mp_hands
except:
    from mediapipe.solutions import hands as mp_hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

dataset_path = "dataset/alphabets"
data, labels = [], []

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path): continue
    
    print("Processing:", label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)
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
df.to_csv("landmark_dataset.csv", index=False)
print("Dataset created:", len(df))