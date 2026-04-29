import torch
from PIL import Image
import torchvision.transforms as transforms
import timm

# -------- CONFIG --------
MODEL_PATH = "models/rice_model.pth"
IMAGE_PATH = "test.jpg"
NUM_CLASSES = 3  # change this!
CLASSES = ["Healthy", "Brown Spot", "Leaf Blast"]

# -------- LOAD MODEL --------
model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------- LOAD IMAGE --------
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# -------- PREDICT --------
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

print("🌾 Disease:", CLASSES[predicted.item()])