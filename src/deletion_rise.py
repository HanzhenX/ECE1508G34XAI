import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

# === CONFIGURATION ===
DATASET_DIR = "./data/SHRUNK_PATHMNIST_EXPLAIN"
MODEL_PATH = "./models/RISE_224x224_RESTNET18_epoch_10.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# === 1. Load Model ===
def load_model(path):
    model = resnet18(num_classes=9)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# === 2. Load Dataset ===
def load_dataset(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "SHRUNK_PATHMNIST_EXPLAIN_predictions.csv"))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    data = []
    for fname in df['filename']:
        label = int(fname.split("_")[1])
        path = os.path.join(data_dir, fname)
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        data.append((img_tensor, label, fname))
    return data

# === 3. Generate RISE Masks ===
def generate_masks(N, s, p1, input_size=(224, 224)):
    cell_size = np.ceil(np.array(input_size) / s).astype(int)
    up_size = (s + 1) * cell_size
    masks = np.empty((N, *input_size), dtype=np.float32)
    for i in tqdm(range(N), desc='Generating masks'):
        grid = (np.random.rand(s, s) < p1).astype(np.float32)
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        resized = np.array(Image.fromarray(grid).resize(up_size[::-1], resample=Image.BILINEAR))
        masks[i] = resized[x:x + input_size[0], y:y + input_size[1]]
    return masks.reshape(N, input_size[0], input_size[1], 1)

# === 4. RISE Saliency Computation ===
def get_saliency(model, image, masks, p1=0.5):
    N = masks.shape[0]
    image = image.unsqueeze(0).to(DEVICE)
    mask_tensor = torch.tensor(masks.transpose(0, 3, 1, 2)).to(DEVICE)
    masked = image * mask_tensor
    with torch.no_grad():
        preds = torch.softmax(model(masked), dim=1).cpu().numpy()
    saliency = np.tensordot(preds.T, masks.squeeze(-1), axes=((1), (0))) / (N * p1)
    return saliency

# === 5. Run Deletion Experiment ===
def run_deletion(model, data, masks, steps):
    os.makedirs("./deletion_results/rise_deletion", exist_ok=True)
    for img_tensor, label, fname in tqdm(data, desc="Running deletion"):
        saliency = get_saliency(model, img_tensor, masks)
        sal_map = saliency[label]
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
        h, w = sal_map.shape
        flat_idx = np.argsort(-sal_map.flatten())
        unnorm = img_tensor * 0.5 + 0.5
        img_clone = unnorm.clamp(0, 1).cpu().numpy()
        scores = []

        for i in range(steps + 1):
            frac = i / steps
            mask = np.ones(h * w, dtype=np.float32)
            mask[flat_idx[:int(frac * h * w)]] = 0
            mask = mask.reshape(h, w)
            masked_img = img_clone * mask[np.newaxis, :, :]
            input_tensor = torch.tensor(masked_img).unsqueeze(0).to(DEVICE)
            input_tensor = (input_tensor - 0.5) / 0.5
            with torch.no_grad():
                prob = torch.softmax(model(input_tensor), dim=1)[0, label].item()
            scores.append(prob)

        plt.plot(np.linspace(0, 1, steps + 1), scores)
        plt.title(f"Rise Deletion Curve: {fname}")
        plt.xlabel("Fraction of Pixels Deleted")
        plt.ylabel("Predicted Probability")
        plt.grid(True)
        plt.ylim(0, 1.0)
        plt.savefig(f"./results/rise_deletion/deletion_{fname}.png", bbox_inches='tight')
        plt.close()

# === Run All ===
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    data = load_dataset(DATASET_DIR)
    masks = generate_masks(N=4000, s=20, p1=0.5)
    run_deletion(model, data, masks, steps=100)
