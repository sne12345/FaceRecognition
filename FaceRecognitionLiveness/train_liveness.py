# USAGE
# python train_liveness.py --dataset dataset --model liveness.model --le le.pickle

import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# 학습 할 초기 학습 속도, 배치 크기 및 에포크 수를 초기화
INIT_LR = 1e-4
BS = 8
EPOCHS = 50

# 데이터 목록 로딩 후 변수 초기화
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
	# 파일 이름에서 클래스 레이블을 추출하고 이미지를 로드한 다음, 32x32 크기 조정
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))

	# 데이터 및 라벨 목록을 각각 업데이트
	data.append(image)
	labels.append(label)

# 데이터를 NumPy 배열로 변환 한 다음 모든 픽셀 강도를 [0, 1] 범위로 스케일링하여 전처리
data = np.array(data, dtype="float") / 255.0

# 레이블을 정수로 인코딩 한 다음 One-Hot 인코딩
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# 훈련용 데이터와 테스트 데이터의 분할을 75:25로 분할
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# 데이터 확장 위한 학습 이미지 생성기 구성
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# 옵티마이저 및 모델 초기화
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# 딥러닝 학습
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# 학습 평가
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# 학습 결과 저장
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"], save_format="h5")

# 레이블 인코더 저장
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# 훈련 손실도 및 정확도 Plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])