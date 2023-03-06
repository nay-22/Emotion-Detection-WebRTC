from pydantic import BaseModel
from aiortc import MediaStreamTrack
from keras.models import model_from_json
import cv2
import numpy as np
from av import VideoFrame


class Offer(BaseModel):
    sdp: str
    type: str
    video_transform: str


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, transform, emotion_model, face_detector):
        super().__init__()
        self.track = track
        self.transform = transform
        self.emotion_dict = {0: "0: Angry", 1: "1: Disgusted", 2: "2: Fearful", 3: "3: Happy", 4: "4: Neutral", 5: "5: Sad", 6: "6: Surprised"}
        self.face_detector = face_detector
        self.emotion_model = emotion_model

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == 'emotion':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = self.face_detector.detectMultiScale(gray_frame, scaleFactor=1, minNeighbhors=5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y+h, x:x+w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion = self.emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion))
                maxval =  "{:.2f}".format(np.amax(emotion)*100)

                cv2.putText(frame, self.emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, str(maxval)+"%", (x+200, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            return frame

        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            img = cv2.bitwise_and(img_color, img_edges)

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        elif self.transform == "edges":

            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        elif self.transform == "rotate":

            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D(
                (cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        return frame
