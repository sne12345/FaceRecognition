from pymongo import MongoClient
from PIL import Image
import io

client = MongoClient('mongodb+srv://naeun:naeuntest@boilerplate.agjuj.mongodb.net/myFirstDatabase?ssl=true&ssl_cert_reqs=CERT_NONE')

db = client.get_database('myFirstDatabase')
records = db.images

print(records.count_documents({}))
data = list(db.images.find())[0]['img'][0]['image']


image = Image.open(io.BytesIO(data), mode='r')
image.save('save1.png')