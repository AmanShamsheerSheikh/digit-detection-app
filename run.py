import torch
import torchvision
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import lenet

lenet = lenet()
lenet.load_state_dict(torch.load("lenet.pth",map_location=torch.device('cpu')))
lenet.eval()

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
def inference_img(img):
    # x = (255 - np.expand_dims(np.array(img), -1))/255.
    print(img.shape)
    with torch.no_grad():
        pred = lenet(torch.unsqueeze(T(img), axis=0).float())
        ans = np.argmax(F.softmax(pred, dim=-1).numpy())
        print(ans)
        return np.argmax(F.softmax(pred, dim=-1).numpy()).item()

# path = r"C:\Users\Aman Sheikh\Downloads\1h.jpeg"
# pred = inference_img(path, lenet)
# pred_idx = np.argmax(pred)
# print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx]*100} %")
