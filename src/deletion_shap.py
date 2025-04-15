import os
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import resnet18
from skimage.segmentation import slic

# === CONFIGURATION ===
DATASET_DIR = "./data/SHRUNK_PATHMNIST_EXPLAIN"
MODEL_PATH = "./models/RISE_224x224_RESTNET18_epoch_10.pt"
RESULTS_DIR = "./deletion_results/shap_deletion"
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

# === SHAP Wrapper ===
def predict_wrapper(x, model):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if x_tensor.shape[-1] == 3:
        x_tensor = x_tensor.permute(0, 3, 1, 2)
    x_tensor = (x_tensor - 0.5) / 0.5
    x_tensor = x_tensor.to(DEVICE)
    with torch.no_grad():
        output = model(x_tensor)
    return torch.softmax(output, dim=1).cpu().numpy()

# === SHAP Saliency Map ===
def get_shap_saliency(model, img_tensor, label):
    img = img_tensor * 0.5 + 0.5
    img = img.clamp(0, 1)
    np_img = img.squeeze().permute(1, 2, 0).cpu().numpy()

    masker = shap.maskers.Image("inpaint_telea", np_img.shape)
    explainer = shap.Explainer(
        lambda x: predict_wrapper(x, model),
        masker,
        algorithm="partition",
        segmentation_fn=lambda x: slic(x, n_segments=1000, compactness=20)
    )

    shap_vals = explainer(np_img[np.newaxis], outputs=[label], max_evals=1000, batch_size=20)
    saliency = np.abs(shap_vals.values[0]).sum(axis=(2, 3)).astype(np.float32)
    return saliency

# === Run Deletion Experiment ===
def run_deletion(model, data, steps=100):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for img_tensor, label, fname in tqdm(data, desc="Running SHAP deletion"):
        saliency = get_shap_saliency(model, img_tensor, label)
        h, w = saliency.shape
        flat_idx = np.argsort(-saliency.flatten())
        img_clone = (img_tensor * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

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

        plt.plot(np.linspace(0, 1, steps + 1), scores)
        plt.title(f"SHAP Deletion Curve: {fname}")
        plt.xlabel("Fraction of Pixels Deleted")
        plt.ylabel("Predicted Probability")
        plt.grid(True)
        plt.ylim(0, 1.0)
        plt.savefig(f"{RESULTS_DIR}/deletion_{fname}.png", bbox_inches='tight')
        plt.close()

# === Main ===
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    data = load_dataset(DATASET_DIR)
    run_deletion(model, data, steps=100)