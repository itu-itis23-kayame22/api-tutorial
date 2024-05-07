from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import torch
from arch17 import Model

app = FastAPI()

# Load model
model = Model()
model.load_state_dict(torch.load('model_23.pth'))
model.eval()


class ImageInput(BaseModel):
    image_url: str


class PredictionOutput(BaseModel):
    predictions_array: List[float]


def predict(image_url: str) -> List[float]:
    try:
        
        image = Image.open(image_url)
        preprocess = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        
        with torch.no_grad():
            output = model(input_batch)
        predictions = output.tolist()[0]
        return predictions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/classify", response_model=PredictionOutput)
async def classify_image(input_data: ImageInput):
    predictions = predict(input_data.image_url)
    return {"predictions_array": predictions}
