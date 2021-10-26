from fastapi import FastAPI
from fastapi.datastructures import UploadFile
from fastapi.param_functions import File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

import uvicorn
app = FastAPI()
MODEL = tf.keras.models.load_model("models/1")
class_names = ["Early Blight","Late Blight","Healthy"]
@app.get("/ping")
async def ping():
    return "Hello people"
def read_files_as_image(data) ->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    image = read_files_as_image(await file.read())
    image = np.expand_dims(image,0)
    pred = MODEL.predict(image)
    pred_class = class_names[np.argmax(pred[0])]
    confidence = np.max(pred[0])
    return {
        'Predicted class': pred_class,
        'Confidence':float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app,host='localhost',port = 8001)