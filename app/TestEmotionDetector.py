import cv2
import numpy as np
from keras.models import model_from_json
import time as t
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from pprint import pprint

emotion_dict = {0: "0: Angry", 1: "1: Disgusted", 2: "2: Fearful", 3: "3: Happy", 4: "4: Neutral", 5: "5: Sad", 6: "6: Surprised"}

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

TimeData = []
EmotionIndex = []


def execute_model():
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    time_counter=0
    while True:
    # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        # face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            pprint(emotion_prediction)
            maxindex = int(np.argmax(emotion_prediction))
            maxval =  "{:.2f}".format(np.amax(emotion_prediction)*100)

            TimeData.append(time_counter)
            EmotionIndex.append(maxindex)
            dataframe = pd.DataFrame(list(zip(TimeData,EmotionIndex)), columns=['Time','Emotion'])
            dataframe.to_csv("emotion_captured.csv")
            time_counter=+1

            print(maxindex)
            print(maxval)

            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, str(maxval)+"%", (x+200, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# execute_model()

    cap.release()
    cv2.destroyAllWindows()

# execute_model()