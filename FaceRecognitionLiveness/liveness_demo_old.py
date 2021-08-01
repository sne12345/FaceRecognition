# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

#  tf 버전 호환문제
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 얼굴 탐지기 로딩
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 진짜 얼굴 탐지 모델 및 레이블 로딩
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# 비디오 스트림을 초기화
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 비디오 스트림에서 프레임 반복
while True:
	# 비디오 스트림 프레임의 이미지 크기를 600 픽셀 너비로 조정
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# blob 이미지로 변환
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# blob 이미지를 통해 OpenCV의 딥러닝 기반 얼굴 탐지기 이용하여 탐지 및 예측 진행
	net.setInput(blob)
	detections = net.forward()

	# 탐지 반복
	for i in range(0, detections.shape[2]):
		# 예측과 관련된 신뢰도(확률)를 추출
		confidence = detections[0, 0, i, 2]

		# 최소 확률 감지 임계값과 비교하여 계산된 확률이 최소 확률보다 큰지 확인
		if confidence > args["confidence"]:
			# 얼굴 경계 상자의 (x, y) 좌표를 계산하고 얼굴 ROI 추출
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 감지된 경계 상자가 프레임의 치수를 벗어나지 않도록 주의
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# 얼굴 ROI를 추출한 다음 훈련 데이터와 정확히 동일한 방식으로 선행 처리
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			# 훈련된 진짜 얼굴 탐지기 모델을 통해 얼굴 ROI를 전달하여 얼굴이 진짜인지 가짜인지 확인
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]

			# 프레임의 경계 상자와 레이블 그린다.
			label = "{}: {:.4f}".format(label, preds[j])
			cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

	# 출력 프레임을 표시
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 'q' 키가 입력되면 루프 탈출
	if key == ord("q"):
		break

# Cleaning
cv2.destroyAllWindows()
vs.stop()