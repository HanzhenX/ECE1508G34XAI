import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import resnet18
from lime import lime_image
from skimage.segmentation import felzenszwalb
import csv
from sklearn.metrics import auc

# === CONFIGURATION ===
DATASET_DIR = "./data/SHRUNK_PATHMNIST_EXPLAIN"
MODEL_PATH = "./models/RISE_224x224_RESTNET18_epoch_10.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# === Load Model ===
def load_model(path):
    model = resnet18(num_classes=9)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# === Load Dataset ===
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

# === LIME Prediction Wrapper ===
def predict_fn(images, model):
    model.eval()
    tensor = torch.tensor(
        np.array([img.transpose(2, 0, 1) for img in images]) / 255.0,
        dtype=torch.float32
    )
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
    return probs.cpu().numpy()

# === Compute LIME Saliency ===
def get_lime_saliency(model, np_img, label):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=np_img,
        classifier_fn=lambda imgs: predict_fn(imgs, model),
        top_labels=9,
        num_samples=1000,
        segmentation_fn=lambda x: felzenszwalb(x, scale=30, sigma=0.5, min_size=20)
    )
    _, mask = explanation.get_image_and_mask(
        label=label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    return mask.astype(np.float32)

# === Run Deletion Experiment ===
def run_deletion(model, data, steps=100):
    os.makedirs("./deletion_results", exist_ok=True)
    auc_path = "./deletion_results/lime_auc.csv"
    with open(auc_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "auc"])
        
    for img_tensor, label, fname in tqdm(data, desc="Running LIME deletion"):
        unnorm = img_tensor * 0.5 + 0.5
        np_img = (unnorm.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        sal_map = get_lime_saliency(model, np_img, label)
        h, w = sal_map.shape
        flat_idx = np.argsort(-sal_map.flatten())
        img_clone = np_img.transpose(2, 0, 1) / 255.0
        scores = []

        for i in range(steps + 1):
            frac = i / steps
            mask = np.ones(h * w, dtype=np.float32)
            mask[flat_idx[:int(frac * h * w)]] = 0
            mask = mask.reshape(h, w)
            masked_img = img_clone * mask[np.newaxis, :, :]
            input_tensor = torch.tensor(masked_img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            input_tensor = (input_tensor - 0.5) / 0.5
            with torch.no_grad():
                prob = torch.softmax(model(input_tensor), dim=1)[0, label].item()
            scores.append(prob)

        x_vals = np.linspace(0, 1, steps + 1)
        auc_val = auc(x_vals, scores)
        plt.plot(x_vals, scores)
        plt.title(f"LIME Deletion Curve: {fname}\nAUC = {auc_val:.4f}")
        plt.xlabel("Fraction of Pixels Deleted")
        plt.ylabel("Predicted Probability")
        plt.grid(True)
        plt.ylim(0, 1.0)
        plt.savefig(f"./deletion_results/lime_deletion/deletion_{fname}.png", bbox_inches='tight')
        plt.close()

        with open(auc_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([fname, auc_val])

# === Main ===
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    data = load_dataset(DATASET_DIR)
    run_deletion(model, data, steps=100)
