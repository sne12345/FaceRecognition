# -*- coding: utf-8 -*-

# USAGE
# python ./FaceLandmarks/detect_face_parts.py --shape-predictor ./FaceLandmarks/shape_predictor_68_face_landmarks.dat --image ./FaceLandmarks/images/naeun.JPG

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

	# 얼굴 부분을 개별적으로 반복
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# 원본 이미지를 복제하여 얼굴의 부분에 대한 이름을 표시
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# 얼굴 랜드마크의 하위 세트를 반복하여 특정 얼굴 부분을 그린다
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

		# 얼굴 영역의 ROI를 별도의 이미지로 추출
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

		# 특정 얼굴 부분 보여 줌
		cv2.imshow("ROI", roi)
		cv2.imshow("Image", clone)
		cv2.waitKey(0)

	# 얼굴 랜드마크 부분 시각화
	output = face_utils.visualize_facial_landmarks(image, shape)
	cv2.imshow("Image", output)
	cv2.waitKey(0)