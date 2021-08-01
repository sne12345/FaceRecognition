# -*- coding: utf-8 -*-

# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/jws.jpg

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# dlib의 얼굴 탐지기(HOG 기반)를 초기화 및 얼굴 랜드마크 예측 변수 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 입력 이미지를 로드하여 크기를 조정하고 grayscale로 변환
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# grayscale 이미지에서 얼굴을 감지
rects = detector(gray, 1)

# 얼굴 탐지 반복
for (i, rect) in enumerate(rects):
	# 얼굴 영역의 랜드마크를 결정하고, 얼굴 랜드마크 (x,y)좌표를 NumPy 배열로 변환
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# dlib의 사각형을 OpenCV 스타일 경계 상자로 변환
	# ex) (x, y, w, h) 얼굴 경계 상자를 그림
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# 얼굴 번호 부여
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# 얼굴 랜드 마크의 (x, y) 좌표를 반복하여 이미지에 그린다.
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# 얼굴 인식 + 얼굴 랜드 마크가있는 출력 이미지 표시
cv2.imshow("Output", image)
cv2.waitKey(0)