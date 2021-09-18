# -*- coding: utf-8 -*- 

# USAGE
# python ./FaceRecognitionLiveness/gather_examples.py --input ./FaceRecognitionLiveness/videos/real.mov --output ./FaceRecognitionLiveness/dataset/real --detector ./FaceRecognitionLiveness/face_detector --skip 1
# python ./FaceRecognitionLiveness/gather_examples.py --input ./FaceRecognitionLiveness/videos/fake_NewMask_valid.MOV --output ./FaceRecognitionLiveness/dataset/fake_val --detector ./FaceRecognitionLiveness/face_detector --skip 4

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True, help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16, help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# 얼굴 탐지기 로딩
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 비디오 파일 스트림 초기화
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 154

# 비디오 파일 스트림 프레임 반복
while True:
	# 파일에서 비디오 스트림 프레임 입력
	(grabbed, frame) = vs.read()

	# 더이상 프레임이 없으면 루프 탈출
	if not grabbed:
		break

	# 프레임수 증가
	read += 1

	# 프레임을 처리해야 하는지 확인
	if read % args["skip"] != 0:
		continue

	# 프레임에서 blob 구성
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# 입력된 이미지에서 얼굴을 인식하기 위해 OpenCV의 딥러닝 기반 얼굴 탐지기 이용
	net.setInput(blob)
	detections = net.forward()

	# 적어도 하나의 얼굴이 발견되었는지 확인
	if len(detections) > 0:
		# 각 이미지가 하나의 얼굴만을 가지고 있다고 가정하고, 가장 큰 확률을 가진 경계 상자를 찾음
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# 확률이 가장 큰 탐지는 최소 확률 테스트를 의미
		if confidence > args["confidence"]:
			# 얼굴 경계 상자의 (x,y) 좌표 계산하고 얼굴 ROI 추출
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]

			# 프레임 쓰기
			p = os.path.sep.join([args["output"], "{}.png".format(saved)])
			# print(face)
			cv2.imwrite(p, face)
			saved += 1
			print("[INFO] saved {} to disk".format(p))

# cleaning
vs.release()
cv2.destroyAllWindows()