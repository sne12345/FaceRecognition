# -*- coding: utf-8 -*-

# USAGE
# python ./FaceLandmarks/detect_drowsiness.py --shape-predictor ./FaceLandmarks/shape_predictor_68_face_landmarks.dat
# python ./FaceLandmarks/detect_drowsiness.py --shape-predictor ./FaceLandmarks/shape_predictor_68_face_landmarks.dat --alarm alarm.wav

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# 경보음 사운드 동작
	playsound.playsound(path)

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
ap.add_argument("-a", "--alarm", type=str, default="", help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())
 
# 눈의 종횡비가 깜박임을 나타내는 상수와 임계값 보다 낮아야 하는 연속 프레임 수에 대한 상수 정의
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# 경보음 발생 여부를 나타내는데 사용되는 Bool 뿐만 아니라 프레임 카운터 초기화
COUNTER = 0
ALARM_ON = False

# dlib의 얼굴 탐지기 (HOG 기반)를 초기화 한 다음 얼굴 랜드 마크 예측 변수 생성
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 왼쪽 눈과 오른쪽 눈에 대한 얼굴 랜드 마크의 인덱스 설정
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Video Stream 초기화
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# Video Stream 반복
while True:
	# 스레드 비디오 파일 스트림에서 프레임을 가져 와서 크기를 조정한 다음 Grayscale 채널로 변환
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Grayscale 프레임에서 얼굴 감지
	rects = detector(gray, 0)

	# 얼굴 감지 반복
	for rect in rects:
		# 얼굴 영역의 얼굴 랜드 마크를 결정한 다음 얼굴 랜드 마크 (x, y) 좌표를 NumPy 배열로 변환
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# 왼쪽 및 오른쪽 눈 좌표를 추출한 다음 좌표를 사용하여 두 눈의 눈 종횡비를 계산
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# 두 눈의 평균 눈 종횡비
		ear = (leftEAR + rightEAR) / 2.0

		# 왼쪽 및 오른쪽 눈 좌표를 추출한 다음 좌표를 사용하여 두 눈의 눈 종횡비 계산
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# 눈의 종횡비가 깜박임 임계값 미만인지 확인하고, 그렇다면 눈 깜박임 프레임 카운터를 늘림
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# 눈의 깜박임 수가 연속 깜박임 프레임 임계값 보다 큰 경우 경보음 울림
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# 경보음이 켜져 있지 않으면 켠다
				if not ALARM_ON:
					ALARM_ON = True

					# 경보음 파일이 제공되었는지 확인하고, 경보음 소리가 재생되도록 스레드를 시작
					if args["alarm"] != "":
						t = Thread(target=sound_alarm, args=(args["alarm"],))
						t.deamon = True
						t.start()

				# 프레임에 경보음 표시
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# 그렇지 않으면, 눈 종횡비가 깜박임 임계 값보다 낮지 않으므로 카운터 및 경보음을 재설정
		else:
			COUNTER = 0
			ALARM_ON = False

		# 올바른 눈 종횡비 임계 값 및 프레임 카운터를 디버깅하고 설정하는데 도움이되도록 계산된 눈 종횡비를 프레임에 그린다.
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# 프레임 표시
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# 'q' key 를 누르면 루프 탈출
	if key == ord("q"):
		break

# cleanup
cv2.destroyAllWindows()
vs.stop()