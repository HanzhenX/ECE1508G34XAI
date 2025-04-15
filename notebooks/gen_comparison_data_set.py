import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from medmnist import INFO, PathMNIST
from torch.utils.data import DataLoader

# === CONFIGURATION ===
EXPERIMENT_NAME = "SHRUNK_PATHMNIST_EXPLAIN"
MODEL_PATH = "./models/RISE_224x224_RESTNET18_epoch_10.pt"
SAVE_DIR = f"./data/{EXPERIMENT_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)
# Device setup
# Check for device
if torch.cuda.is_available():
	device = torch.device("cuda")
	print("Using CUDA (GPU)")
	
# Check for MPS (Apple Silicon Macs)
elif torch.backends.mps.is_available():
	device = torch.device("mps")
	print("Using MPS (macOS)")
	
else:
	device = torch.device("cpu")
	print("Using CPU")

# === Load model ===
model = resnet18(num_classes=9)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Load dataset ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

info = INFO['pathmnist']
test_dataset = PathMNIST(split='test', transform=transform, download=True, size=224, mmap_mode = 'r')

# === Select first 100 per class ===
class_counts = {i: 0 for i in range(9)}
filtered_data = []
for img, label in test_dataset:
    label = int(label)
    if class_counts[label] < 100:
        filtered_data.append((img, label))
        class_counts[label] += 1
    if all(v >= 100 for v in class_counts.values()):
        break

# === Prepare result tracking ===
correct_per_class = {i: 0 for i in range(9)}
wrong_per_class = {i: 0 for i in range(9)}
results = []

# === Run inference and save results ===
for idx, (img_tensor, label) in enumerate(filtered_data):
    img_tensor = img_tensor.to(device)
    input_tensor = img_tensor.unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred = int(np.argmax(probs))

    # Check if it’s a needed correct/wrong case
    if pred == label and correct_per_class[label] < 2:
        correct_per_class[label] += 1
        keep = True
    elif pred != label and wrong_per_class[label] < 2:
        wrong_per_class[label] += 1
        keep = True
    else:
        keep = False

    if keep:
        # Save image
        unnorm_img = (img_tensor.cpu() * 0.5 + 0.5).clamp(0, 1).numpy().transpose(1, 2, 0)
        img_uint8 = (unnorm_img * 255).astype(np.uint8)
        filename = f"img{idx}_{label}_{pred}.png"
        Image.fromarray(img_uint8).save(os.path.join(SAVE_DIR, filename))

        # Save prediction info
        row = {"filename": filename}
        row.update({f"class_{i}": probs[i] for i in range(9)})
        results.append(row)

    # Stop when we have 2 correct and 2 wrong per class
    if all(correct_per_class[c] >= 2 and wrong_per_class[c] >= 2 for c in range(9)):
        break

# === Save predictions to CSV ===
df = pd.DataFrame(results)
df.to_csv(os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_predictions.csv"), index=False)
print(f"✅ Saved {len(df)} images and predictions to {SAVE_DIR}")