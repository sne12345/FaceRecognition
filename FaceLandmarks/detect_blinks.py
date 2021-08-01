# -*- coding: utf-8 -*-

# USAGE
# blink_detection_demo.mp4 (자신의 눈 깜박임 영상을 촬영하여 blink_detection_demo.mp4 이름으로 저장 후 실행)
# python ./FaceLandmarks/detect_blinks.py --shape-predictor ./FaceLandmarks/shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python ./FaceLandmarks/detect_blinks.py --shape-predictor ./FaceLandmarks/shape_predictor_68_face_landmarks.dat

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
	# 두 세트의 수직 눈 랜드 마크 (x, y) 좌표 간의 유클리드 거리 계산
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# 수평 눈 랜드 마크 (x, y) 좌표 간의 유클리드 거리 계산
	C = dist.euclidean(eye[0], eye[3])

	# 눈 종횡비 계산
	ear = (A + B) / (2.0 * C)

	# 눈 종횡비 반환
	return ear
 
# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())
 
# 눈의 종횡비가 깜박임을 나타내는 상수와 임계값 보다 낮아야 하는 연속 프레임 수에 대한 상수 정의
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# 프레임 카운터와 총 깜박임 수를 초기화
COUNTER = 0
TOTAL = 0

# dlib의 얼굴 탐지기 (HOG 기반)를 초기화 한 다음 얼굴 랜드 마크 예측 변수 생성
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 왼쪽 눈과 오른쪽 눈에 대한 얼굴 랜드 마크의 인덱스 설정
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Video Stream 초기화
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
fileStream = False
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)

# Video Stream 반복
while True:
	# 파일 비디오 스트림인 경우 처리 할 버퍼에 프레임이 더 남아 있는지 확인
	if fileStream and not vs.more():
		break

	# 스레드 비디오 파일 스트림에서 프레임을 가져 와서 크기를 조정한 다음 Grayscale 채널로 변환
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Grayscale 프레임에서 얼굴 감지
	rects = detector(gray, 0)

	# 얼굴 감지 반복
	for rect in rects:
		# 얼굴 영역의 얼굴 랜드 마크를 결정한 다음 얼굴 랜드 마크 (x, y) 좌표를 NumPy 배열로 변환
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# 왼쪽 및 오른쪽 눈 좌표를 추출한 다음 좌표를 사용하여 두 눈의 눈 종횡비 계산
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# 두 눈의 평균 눈 종횡비
		ear = (leftEAR + rightEAR) / 2.0

		# 왼쪽 눈과 오른쪽 눈의 눈꺼플을 계산 한 다음 각 눈을 시각화
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# 눈의 종횡비가 깜박임 임계값 보다 낮은지 확인하고, 그렇다면 눈 깜박임 프레임 카운터를 늘림
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		# 그렇지 않으면, 눈의 종횡비가 깜박임 임계값 보다 낮지 않음
		else:
			# 눈의 깜박임 수가 연속 깜박임 프레임 임계값 보다 큰 경우 총 깜박임 횟수 증가
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			# 눈 깜박임 프레임 카운터 재설정
			COUNTER = 0

		# 프레임의 계산 된 눈 종횡비와 함께 프레임의 총 깜박임 수 표시
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# 프레임 보여줌
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# 'q' key 를 누르면 루프 탈출
	if key == ord("q"):
		break

# Clean up
cv2.destroyAllWindows()
vs.stop()