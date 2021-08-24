# USAGE
# python ./FaceRecognitionDL/train_model.py --embeddings ./FaceRecognitionDL/output/embeddings.pickle \
# 	--recognizer ./FaceRecognitionDL/output/recognizer.pickle --le ./FaceRecognitionDL/output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# # 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True, help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
args = vars(ap.parse_args())

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