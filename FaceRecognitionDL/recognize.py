# USAGE
# python ./FaceRecognitionDL/recognize.py --detector ./FaceRecognitionDL/face_detection_model \
# 	--embedding_model ./FaceRecognitionDL/openface_nn4.small2.v1.t7 \
# 	--recognizer ./FaceRecognitionDL/output/recognizer.pickle \
# 	--le ./FaceRecognitionDL/output/le.pickle --image ./FaceRecognitionDL/images/patrick_bateman.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-d", "--detector", required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding_model", required=True, help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True, help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 얼굴 탐지기 로딩
# 얼굴을 감지하기 위해 OpenCV에서 제공하는 사전 훈련된 Caffe 딥러닝 모델
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 얼굴 인식기 로딩 (128-D 얼굴 인식을 계산하기 위해 사전 훈련된 Torch DL 모델)
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# 레이블 인코더와 함께 실제 얼굴 인식 모델 로딩 (Linear SVM얼굴 인식모델)
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# 이미지를 로드하고 가로x세로 비율 유지하며 너비가 600 픽셀이 되도록 크기 조정된 이미지 획득
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# 이미지에서 blob 구성
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# 입력된 이미지에서 얼굴을 인식하기 위해 OpenCV의 딥러닝 기반 얼굴 탐지기 이용
detector.setInput(imageBlob)
detections = detector.forward()

# 탐지 반복
for i in range(0, detections.shape[2]):
	# 예측과 관련된 신뢰도(즉, 확률)를 추출
	confidence = detections[0, 0, i, 2]

	# 최소 확률 감지 임계값과 비교하여 계산된 확률이 최소 확률보다 큰지 확인
	if confidence > args["confidence"]:
		# 얼굴 경계 상자의 (x, y) 좌표 계산
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# 얼굴 ROI 추출
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# 얼굴의 너비와 높이가 충분히 큰지 확인
		if fW < 20 or fH < 20:
			continue

		# 얼굴 ROI에 대한 blob을 구성한 다음 얼굴 임베딩 모델을 통해 blob을 전달하여 얼굴의 128-D 벡터 생성
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		# 벡터를 SVM 인식기 모델을 통해 전달
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# 얼굴을 인식하기 위해 분류를 수행 (가장 높은 확률 지수를 취하고 이름을 찾기 위해 레이블 인코더 색인)
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# 관련 확률과 함께 얼굴의 경계 상자를 그립
		text = "{}: {:.2f}%".format(name, proba * 100)
		print(name, proba * 100)

		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# 출력 이미지 보여줌
cv2.imshow("Image", image)
cv2.waitKey(0)