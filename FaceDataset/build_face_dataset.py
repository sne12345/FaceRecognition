# -*- coding: utf-8 -*-

# USAGE
# python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/sunny


# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help = "path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())

# 얼굴 탐지 위한 OpenCV의 Haar cascade 로드
detector = cv2.CascadeClassifier(args["cascade"])

# 비디오 스트림 초기화
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

# 비디오 스트림 프레임 반복
while True:
	# 비디오 스트림에서 프레임을 읽어 복제하고, 프레임 크기 조절
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# 회색조 프레임에서 얼굴 감지
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

	# 얼굴 인식을 반복하고 그것들을 프레임에 그린다.
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# 출력 프레임 Show
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# 'c' Key를 누르면  orig 프레임의 이미지 저장
	if key == ord("c"):
		p = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1

	# 'q' Key를 누르면 루프에서 빠져 나옴
	elif key == ord("q"):
		break

# Cleaning
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()