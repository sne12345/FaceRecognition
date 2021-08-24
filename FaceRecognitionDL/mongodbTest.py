from pymongo import MongoClient
from bson.objectid import ObjectId
from PIL import Image
import io
import os

# 몽고디비 연결
client = MongoClient('mongodb+srv://naeun:naeuntest@boilerplate.agjuj.mongodb.net/myFirstDatabase?ssl=true&ssl_cert_reqs=CERT_NONE')

# myFirstDatabase 데이터베이스 가져오기, image 테이블 가져오기
db = client.get_database('myFirstDatabase')
records = db.images


# 총 데이터수만큼(사람수) for문
for i in range(records.count_documents({})):

    label_str = str(i)

    # 사진수만큼 for문
    for j in range(len(list(db.images.find())[i]['img'])):

        # image byte 데이터 가져오기
        data = list(db.images.find())[i]['img'][j]['image']

        # byte => image
        image = Image.open(io.BytesIO(data), mode='r')

        # 현재 디렉토리 경로
        working_dir = os.getcwd()

        # 경로에 label 파일이 없으면 생성
        if not os.path.exists(working_dir+'/FaceRecognitionDL/dataset/'+label_str):
                    os.makedirs(working_dir+'/FaceRecognitionDL/dataset/'+label_str)

        # 이미지 저장
        idx_str = str(j)
        image.save(f'./FaceRecognitionDL/dataset/{label_str}/image_{idx_str}.png')