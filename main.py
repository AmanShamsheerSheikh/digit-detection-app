from run import inference_img
from fastapi import FastAPI,UploadFile
from PIL import Image
import io
import numpy as np
import torch
from torchvision.transforms import ToTensor,Resize
import uvicorn
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/send")
def send(img:UploadFile):
    content = img.file.read()
    img = Image.open(io.BytesIO(content)).convert(mode="L")
    img = img.resize((28, 28))
    x = (255 - np.expand_dims(np.array(img), -1))/255.
    # image = Image.open(io.BytesIO(content)).convert(mode="L")
    # transform = Resize((28,28))
    # image = transform(image)
    # image = ToTensor()(image)
    result = inference_img(x)
    return result

# if __name__ == "__main__":
#     uvicorn.run(app, port=8000,reload=True)