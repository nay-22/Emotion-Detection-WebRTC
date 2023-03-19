import cv2 as cv
import numpy as np

video = cv.VideoCapture(0)

while True:
    _, frame = video.read()

    img = frame.to_ndarray(format='bgr24')
    print(img)

    # cv.imshow('Test', img)

    if cv.waitKey(20) and 0xFF == ord('d'):
        break

video.release()
cv.destroyAllWindows()