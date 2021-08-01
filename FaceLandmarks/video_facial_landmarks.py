# -*- coding: utf-8 -*-

# USAGE
# python ./FaceLandmarks/video_facial_landmarks.py --shape-predictor ./FaceLandmarks/shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
 
# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# dlib의 얼굴 탐지기(HOG 기반)를 초기화 및 얼굴 랜드마크 예측 변수 생성
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 비디오 스트림을 초기화
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# 비디오 스트림 프레임 반복
while True:
	# 비디오 스트림에서 프레임을 잡아 크기를 400픽셀이 되도록 조정, Grayscale로 변환
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Grayscale 에서 프레임 얼굴 감지
	rects = detector(gray, 0)

	# 얼굴 인식 반복
	for rect in rects:
		# 얼굴 영역의 랜드마크를 결정하고, 얼굴 랜드마크 (x,y)좌표를 NumPy 배열로 변환
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# 얼굴 랜드마크의 하위 세트를 반복하여 특정 얼굴 부분을 그린다
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	# 프레임을 보여줌
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# 'q' key 를 누르면 루프 탈출
	if key == ord("q"):
		break
 
# Clean up
cv2.destroyAllWindows()
vs.stop()