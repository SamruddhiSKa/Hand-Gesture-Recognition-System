import cv2
import os
import time

cap = cv2.VideoCapture(0)

while True:
    label = input("\nEnter alphabet (or exit): ")  
    if label == "exit": break

    path = f"dataset/gestures/{label}"
    os.makedirs(path, exist_ok=True)
    count = len(os.listdir(path))

    print("Starting capture in 3 seconds...")
    time.sleep(3)

    start_time = time.time()
    duration = 15

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame,(200,100),(400,300),(0,255,0),2)
        roi = frame[100:300, 200:400]
        cv2.imshow("Frame", frame)
        cv2.imwrite(f"{path}/{count}.jpg", roi)
        count += 1

        if time.time() - start_time > duration: break
        if cv2.waitKey(150) == 27: break

cap.release()
cv2.destroyAllWindows()