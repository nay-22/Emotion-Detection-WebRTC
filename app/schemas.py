from pydantic import BaseModel
from aiortc import MediaStreamTrack
from keras.models import model_from_json
from fastapi import HTTPException
import cv2 as cv
import numpy as np
from av import VideoFrame


class Offer(BaseModel):
    sdp: str
    type: str
    video_transform: str


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, transform, face_detector, emotion_model):
        super().__init__()
        self.track = track
        self.transform = transform
        self.emotion_dict = {0: "0: Angry", 1: "1: Disgusted", 2: "2: Fearful", 3: "3: Happy", 4: "4: Neutral", 5: "5: Sad", 6: "6: Surprised"}
        self.face_detector = face_detector
        self.emotion_model = emotion_model

    async def recv(self):
        frame = await self.track.recv()

        if self.transform =='test':
            img = frame.to_ndarray(format='bgr24')
            cv.rectangle(img, (100,100), (200,200), (0,0,255), 4)
            nframe = VideoFrame.from_ndarray(img, format='bgr24')
            nframe.pts = frame.pts
            nframe.time_base = frame.time_base
            return nframe

        elif self.transform == 'emotion':
            img = frame.to_ndarray(format='bgr24')
            gray = frame.to_ndarray(format='gray')

            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x,y,w,h) in faces:
                cv.rectangle(img, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray_frame, (48, 48)), -1), 0)

                # emotion_prediction = self.emotion_model.predict(cropped_img)
                # maxindex = int(np.argmax(emotion_prediction))
                # maxval =  "{:.2f}".format(np.amax(emotion_prediction)*100)

                # print(maxindex)
                # print(maxval)

                # cv.putText(img, self.emotion_dict[maxindex], (x+5, y-20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                # cv.putText(img, str(maxval)+"%", (x+200, y-20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

            nframe = VideoFrame.from_ndarray(img, format="bgr24")
            nframe.pts = frame.pts
            nframe.time_base = frame.time_base
            return nframe

        elif self.transform == "edges":

            img = frame.to_ndarray(format="bgr24")
            # print(img)
            img = cv.cvtColor(cv.Canny(img, 100, 200), cv.COLOR_GRAY2BGR)

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        elif self.transform == "rotate":

            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv.getRotationMatrix2D(
                (cols / 2, rows / 2), frame.time * 45, 1)
            img = cv.warpAffine(img, M, (cols, rows))

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv.pyrDown(cv.pyrDown(img))
            for _ in range(6):
                img_color = cv.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv.pyrUp(cv.pyrUp(img_color))

            # prepare edges
            img_edges = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            img_edges = cv.adaptiveThreshold(
                cv.medianBlur(img_edges, 7),
                255,
                cv.ADAPTIVE_THRESH_MEAN_C,
                cv.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv.cvtColor(img_edges, cv.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        return frame
