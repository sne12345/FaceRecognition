# USAGE
# python ./FaceRecognitionLiveness/train_liveness.py --dataset ./FaceRecognitionLiveness/dataset --model ./FaceRecognitionLiveness/liveness.model --le ./FaceRecognitionLiveness/le.pickle


import cv2
import inspect
import tensorflow as tf
from keras import Sequential, applications
from keras.callbacks import Callback, EarlyStopping
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import matplotlib.pyplot as plt
import random
from timeit import default_timer as timer
from tqdm import tqdm
import argparse

# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


# 마스크 착용된 사진들의 디렉토리 모음
with_mask_dirs = ['/opt/ml/kaggle/input/face-mask-12k-images-dataset/Train/WithMask', 
                  '/opt/ml/kaggle/input/face-mask-12k-images-dataset/Validation/WithMask',
                  '/opt/ml/kaggle/input/face-mask-12k-images-dataset/Test/WithMask',
                  '/opt/ml/kaggle/input/covid-face-mask-detection-dataset/Train/Mask',
                  '/opt/ml/kaggle/input/covid-face-mask-detection-dataset/Validation/Mask',
                  '/opt/ml/kaggle/input/covid-face-mask-detection-dataset/Test/Mask']
# 마스크 미착용된 사진들의 디렉토리 모음
without_mask_dirs = ['/opt/ml/kaggle/input/face-mask-12k-images-dataset/Train/WithoutMask',
                     '/opt/ml/kaggle/input/face-mask-12k-images-dataset/Validation/WithoutMask',
                     '/opt/ml/kaggle/input/face-mask-12k-images-dataset/Test/WithoutMask',
                     '/opt/ml/kaggle/input/covid-face-mask-detection-dataset/Train/Non Mask',
                     '/opt/ml/kaggle/input/covid-face-mask-detection-dataset/Validation/Non Mask',
                     '/opt/ml/kaggle/input/covid-face-mask-detection-dataset/Test/Non Mask']

# 사진들의 디렉토리와 타겟 값을 dataframe으로 묶음
def folder_to_df(dirs, labels):
    file_list = []
    num_folders = len(dirs)
    for count, folder in enumerate(dirs, start=1):
        for file in sorted(os.listdir(folder)):
            if count < num_folders / 2:
                file_list.append((folder + '/' + file, labels[0]))
            else:
                file_list.append((folder + '/' + file, labels[1]))
    return pd.DataFrame(file_list, columns=['filename', 'class'])

labels = ['With Mask', 'Without Mask'] # 타겟 값: 마스크 착용, 마스크 미착용
all_mask_df = folder_to_df(with_mask_dirs + without_mask_dirs, labels) # 모든 사진들의 디렉토리를 dataframe으로 합침

shuffled_mask_df = all_mask_df.sample(frac=1) # 고른 선택을 위해 데이터 전체의 순서를 섞음

train_df, val_df, test_df = np.split(shuffled_mask_df, 
                                     [int(0.8*len(shuffled_mask_df)), int(0.9*len(shuffled_mask_df))]) # 데이터를 80:10:10의 비율로 train, validation, test 분할

# train, validation, test 세트 확인
print('total dataset : {}\ntraining set  : {}\nvalidation set: {}\ntest set      : {}'.format(all_mask_df.shape, train_df.shape, val_df.shape, test_df.shape))
display(train_df.head(5), val_df.head(5), test_df.head(5))

train_datagen = ImageDataGenerator(rotation_range=10, # 기울어진 얼굴
                                   zoom_range=0.1, # 확대된 얼굴
                                   horizontal_flip=True, # 좌우 반전된 얼굴
                                   rescale=1.0/255) # 255개의 RGB values에 맞춰서 축소

val_test_datagen = ImageDataGenerator(rescale=1.0/255) # validation

VAL_BATCH_SIZE = 128
BATCH_SIZE = VAL_BATCH_SIZE * 8

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    target_size=(128,128), # 최대 입력 이미지 크기에 맞춤
    class_mode='binary',
    batch_size=BATCH_SIZE) # 출력 값 2가지: 마스크 착용, 마스크 미착용
val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    target_size=(128,128),
    class_mode='binary',
    batch_size=VAL_BATCH_SIZE)
test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    target_size=(128,128),
    class_mode='binary',
    batch_size=VAL_BATCH_SIZE)


pretrained_base = tf.keras.applications.DenseNet121(
    include_top=False, # 128x128의 이미지 크기를 맞추기 위해 False로 설정
    input_shape=(128,128,3), # 이미지 크기와 RGB 값 갯수
    pooling='avg') 
pretrained_base.trainable = False

model = Sequential([
    pretrained_base,
    Flatten(),
    Dense(1, activation='sigmoid')])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss', 
    min_delta=0,
    patience=10, # 10번의 epoch 동안 val_loss이 더 이상 줄어들지 않을 때 훈련을 멈춤
    verbose=1,
    mode='min',
    restore_best_weights=True)

EPOCHS = 100

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=val_generator)

print(model.evaluate(test_generator))

model.save(args["model"], save_format="h5")