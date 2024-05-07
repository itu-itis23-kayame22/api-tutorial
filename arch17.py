import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import requests
from io import BytesIO
import torchvision.models as models
import torch.nn as nn

image_url = input("Enter URL: ")

model_path = 'model_23.pth'

classes = [
    "Mantled Howler",
    "Patas Monkey",
    "Bald Uakari",
    "Japanese Macaque",
    "Pygmy Marmoset",
    "White Headed Capuchin",
    "Silvery Marmoset",
    "Common Squirrel Monkey",
    "Black Headed Night Monkey",
    "Nilgiri Langur",
]

def initialize_model(num_classes=10): # this needs for architect
    model = models.resnet18(pretrained=False)  # pretrained=False since we're loading our own weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  
    return model

model = initialize_model()

# Load the state dictionary
state_dict = torch.load(model_path)

# Load the state dictionary into the model
model.load_state_dict(state_dict)
  
model.eval()

mean = [0.4363, 0.4328, 0.3291] #this values essential for normalizing images before they are fed into a neural network
std = [0.2129, 0.2075, 0.2038]

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])


def classify(model, image_transforms, image_url, classes, topk=5):
    model = model.eval()  # Ensure the model is in evaluation mode.
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raises an HTTPError if the response is not 200.
        image = Image.open(BytesIO(response.content))

        # Ensure the loaded image is in RGB format.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image_transforms(image).unsqueeze(0)  # Apply transformations

        output = model(image)
        # Get the top k predictions and their indices
        probs, indices = torch.topk(output, topk)
        probs = torch.nn.functional.softmax(probs, dim=1)  # Convert to probabilities

        for i in range(topk):
            print(f"{i+1}: {classes[indices[0][i].item()]} with a probability of {probs[0][i].item()*100:.2f}%")

    except requests.exceptions.RequestException as e:
        print("Error downloading the image:", e)
    except Exception as e:
        print("Error processing the image:", e)


classify(model, image_transforms, image_url, classes, topk=5)