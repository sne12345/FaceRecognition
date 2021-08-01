# USAGE
# python ./FaceRecognitionDL/register_train.py --dataset ./FaceRecognitionDL/dataset --embeddings ./FaceRecognitionDL/output/embeddings.pickle \
#        --detector ./FaceRecognitionDL/face_detection_model --embedding_model ./FaceRecognitionDL/openface_nn4.small2.v1.t7 \
# 	--recognizer ./FaceRecognitionDL/output/recognizer.pickle --le ./FaceRecognitionDL/output/le.pickle


##########################################################################################

# 0. import

##########################################################################################

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import sklearn



##########################################################################################

# 1. extract_embeddings

##########################################################################################


# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True, help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding_model", required=True, help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
args = vars(ap.parse_args())

# 얼굴 탐지기 로딩
# 얼굴을 감지하기 위해 OpenCV에서 제공하는 사전 훈련된 Caffe 딥러닝 모델
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 얼굴 인식기 파일 로딩 (128D 얼굴 임베딩을 생성하는 토치 딥 러닝 모델)
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# 데이터셋 입력 이미지 목록에 대한 경로
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# 추출된 얼굴 및 해당 인물 이름 목록 변수 초기화
knownEmbeddings = []
knownNames = []

# 처리된 총 얼굴의 수 초기화
total = 0

# 입력된 이미지 반복
for (i, imagePath) in enumerate(imagePaths):
	# 이미지 경로에서 사람 이름 추출
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# 이미지를 로드하고 가로x세로 비율 유지하며 너비가 600 픽셀이 되도록 크기 조정된 이미지 획득
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# 이미지에서 blob 구성
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# 입력된 이미지에서 얼굴을 인식하기 위해 OpenCV의 딥러닝 기반 얼굴 탐지기 이용
	detector.setInput(imageBlob)
	detections = detector.forward()

	# 적어도 하나의 얼굴이 발견되었는지 확인
	if len(detections) > 0:
		# 각 이미지가 하나의 얼굴만을 가지고 있다고 가정하고, 가장 큰 확률을 가진 경계 상자를 찾음
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# 확률이 가장 큰 탐지는 최소 확률 테스트를 의미
		if confidence > args["confidence"]:
			# 얼굴 경계 상자의 (x,y) 좌표 계산
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 얼굴 ROI를 추출하고 ROI 치수를 가져옴
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# 얼굴 너비와 높이가 충분히 큰지 확인
			if fW < 20 or fH < 20:
				continue

			# 얼굴 ROI에 대한 blob을 구성한 다음 얼굴이 포함된 모델을 통해 blob을 전달하여 얼굴의 128-d 정량화를 얻음
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# 사람 이름 + 해당 목록에 포함된 얼굴 추가
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# 얼굴 포함 및 이름을 파일에 추가
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()





##########################################################################################

# 2. train_model

##########################################################################################

# # 얼굴 임베딩 로딩
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# # 레이블 인코딩
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# # SVM 모델을 초기화 하고, 얼굴의 128-d 임베딩을 받아들이는데 사용된 모델을 훈련 시킨 후 실제 얼굴 인식
# # 모델 SVM 말고 다른 걸로?
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# # 실제 얼굴인식 모델 파일 쓰기
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# # 라벨 인코딩 쓰기
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

