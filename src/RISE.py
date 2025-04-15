# %% [markdown]
# This is a modified notebook for the project, from the sample code downloaded originally from MedMNIST example at https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb .

# %%
!pip install medmnist

# %%
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Device setup
# Check for CUDA
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

# Example usage: Move model and tensors to the selected device

import medmnist
from medmnist import INFO, Evaluator

# %%
print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# %% [markdown]
# # We first work on a 2D dataset with size 28x28

# %%
data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 30
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# %% [markdown]
# ## First, we read the MedMNIST data, preprocess them and encapsulate them into dataloader form.

# %%
# preprocessing
data_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# %%
import os
import torch

os.makedirs('./xai_data_28x28', exist_ok=True)

class_counts = {i: 0 for i in range(9)}  # 9 classes in PathMNIST
max_per_class = 2

for images, labels in train_loader:
    for img_tensor, label in zip(images, labels):
        label = int(label)
        if class_counts[label] >= max_per_class:
            continue

        save_path = f'./xai_data_28x28/class_{label}_{class_counts[label]}.pt'
        torch.save(img_tensor, save_path)
        class_counts[label] += 1

    if all(c >= max_per_class for c in class_counts.values()):
        break

print("Saved 2 tensor samples per class to ./xai_data_28x28/")

# %%
## Dataloader to load the saved tensor instead for explainability experiments.
## Usage: 
# from torch.utils.data import DataLoader

# xai_dataset = SavedTensorDataset('./xai_data_28x28') # or xai_data_224 for the resnet experiment
# xai_loader = DataLoader(xai_dataset, batch_size=1, shuffle=False)

# for img_tensor, label in xai_loader:
#     # Use `img_tensor` for inference, saliency, etc.
#     print(img_tensor.shape, label.item())

import os
import torch
from torch.utils.data import Dataset

class SavedTensorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted([
            f for f in os.listdir(root_dir) if f.endswith(".pt")
        ])
        self.labels = [int(f.split('_')[1]) for f in self.file_list]  # from filename

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        tensor = torch.load(file_path)
        label = self.labels[idx]
        return tensor, label

# %%
print(train_dataset)
print("===================")
print(test_dataset)

# %%


# %%
# visualization

train_dataset.montage(length=1)

# %%
# montage

train_dataset.montage(length=20)

# %% [markdown]
# ## Then, we define a simple model for illustration, object function and optimizer that we use to classify.

# %%
# define a simple CNN model

class Net(nn.Module):
	def __init__(self, in_channels, num_classes):
		super(Net, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=3),
			nn.BatchNorm2d(16),
			nn.ReLU())

		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 16, kernel_size=3),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))

		self.layer3 = nn.Sequential(
			nn.Conv2d(16, 64, kernel_size=3),
			nn.BatchNorm2d(64),
			nn.ReLU())
		
		self.layer4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3),
			nn.BatchNorm2d(64),
			nn.ReLU())

		self.layer5 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))

		self.fc = nn.Sequential(
			nn.Linear(64 * 4 * 4, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes))

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

model = Net(in_channels=n_channels, num_classes=n_classes).to(device)
	
# define loss function and optimizer
if task == "multi-label, binary-class":
	criterion = nn.BCEWithLogitsLoss()
else:
	criterion = nn.CrossEntropyLoss()
	
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# %% [markdown]
# ## Next, we can start to train and evaluate!

# %%
# evaluation

def test(split):
	model.eval()
	y_true = torch.tensor([], device=device)
	y_score = torch.tensor([], device=device)
	
	data_loader = train_loader_at_eval if split == 'train' else test_loader

	with torch.no_grad():
		for inputs, targets in data_loader:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)

			if task == 'multi-label, binary-class':
				targets = targets.to(torch.float32)
				outputs = outputs.softmax(dim=-1)
			else:
				targets = targets.squeeze().long()
				outputs = outputs.softmax(dim=-1)
				targets = targets.float().resize_(len(targets), 1)

			y_true = torch.cat((y_true, targets), 0)
			y_score = torch.cat((y_score, outputs), 0)

		y_true = y_true.cpu().numpy()
		y_score = y_score.cpu().detach().numpy()
		
		evaluator = Evaluator(data_flag, split)
		metrics = evaluator.evaluate(y_score)
	
		print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

	return metrics

		
print('==> Evaluating ...')
test('train')
test('test')

# %%
# Train and evaluate
EXPERIMENT_NAME = "RISE_28x28_CNN"

def save_current_model(n_epoch):
	save_path = Path(f'./models/{EXPERIMENT_NAME}_epoch_{n_epoch}.pt')
	save_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), save_path)
	print(f"Model saved to {save_path}")

save_current_model(0)

# === Tracking variables ===
train_losses = []
train_accs = []
test_accs = []

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for inputs, targets in tqdm(train_loader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)

		if task == 'multi-label, binary-class':
			targets = targets.to(torch.float32)
			loss = criterion(outputs, targets)
		else:
			targets = targets.squeeze().long()
			loss = criterion(outputs, targets)
			_, predicted = outputs.max(1)
			correct += predicted.eq(targets).sum().item()
			total += targets.size(0)

		loss.backward()
		optimizer.step()
		running_loss += loss.item()

	# Store training metrics
	train_losses.append(running_loss / len(train_loader))
	if task != 'multi-label, binary-class':
		train_acc = correct / total
		train_accs.append(train_acc)
	else:
		train_metrics = test('train')  # for multi-label case, get acc from test function
		train_accs.append(train_metrics[1])  # assuming acc is at index 1

	# Evaluate on test set
	test_metrics = test('test')
	test_accs.append(test_metrics[1])  # assuming acc is at index 1

	print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
		  f"Loss: {train_losses[-1]:.4f} "
		  f"Train Acc: {train_accs[-1]:.4f} "
		  f"Test Acc: {test_accs[-1]:.4f}")
	
	# === Save model ===
	save_current_model(epoch+1)

# === Plotting ===
epochs = range(1, NUM_EPOCHS + 1)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot training loss on left y-axis
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
l1, = ax1.plot(epochs, train_losses, color='tab:red', label='Train Loss')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a second y-axis for accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:blue')
l2, = ax2.plot(epochs, train_accs, color='tab:blue', linestyle='--', label='Train Acc')
l3, = ax2.plot(epochs, test_accs, color='tab:green', linestyle='-', label='Test Acc')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Combine legends
lines = [l1, l2, l3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Training Progress')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%


# %% [markdown]
# # We then check a 2D dataset with size 224x224

# %%
data_flag = 'pathmnist'
download = True

info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download, size=224, mmap_mode='r')
train_dataset = DataClass(split='train', transform=data_transform, download=download, size=224, mmap_mode='r')
test_dataset = DataClass(split='test', transform=data_transform, download=download, size=224, mmap_mode='r')

# encapsulate data into dataloader form
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
# test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# %%
# import os
# import torch

# os.makedirs('./xai_data_224', exist_ok=True)

# class_counts = {i: 0 for i in range(9)}  # 9 classes in PathMNIST
# max_per_class = 2

# for images, labels in train_loader:
#     for img_tensor, label in zip(images, labels):
#         label = int(label)
#         if class_counts[label] >= max_per_class:
#             continue

#         save_path = f'./xai_data_224/class_{label}_{class_counts[label]}.pt'
#         torch.save(img_tensor, save_path)
#         class_counts[label] += 1

#     if all(c >= max_per_class for c in class_counts.values()):
#         break

# print("Saved 2 tensor samples per class to ./xai_data_224/")

# %%
print(train_dataset)
print("===================")
print(test_dataset)

# %%
x, y = train_dataset[0]

print(x.shape, y.shape)

# %%
train_dataset.montage(length=3)

# %% [markdown]
# ## Then we train and evaluate on this 224x224 dataset

# %%
from torchvision.models import resnet18

model = resnet18(num_classes=n_classes).to(device)

criterion = nn.CrossEntropyLoss()
	
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# %%
# evaluation (on initial model)

def test(split):
	model.eval()
	y_true = torch.tensor([], device=device)
	y_score = torch.tensor([], device=device)
	
	data_loader = train_loader_at_eval if split == 'train' else test_loader

	with torch.no_grad():
		for inputs, targets in data_loader:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)

			if task == 'multi-label, binary-class':
				targets = targets.to(torch.float32)
				outputs = outputs.softmax(dim=-1)
			else:
				targets = targets.squeeze().long()
				outputs = outputs.softmax(dim=-1)
				targets = targets.float().resize_(len(targets), 1)

			y_true = torch.cat((y_true, targets), 0)
			y_score = torch.cat((y_score, outputs), 0)

		y_true = y_true.cpu().numpy()
		y_score = y_score.cpu().detach().numpy()
		
		evaluator = Evaluator(data_flag, split)
		metrics = evaluator.evaluate(y_score)
	
		print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
		return metrics

		
print('==> Evaluating ...')
test('train')
test('test')

# %%
# Train and evaluate
EXPERIMENT_NAME = "RISE_224x224_RESTNET18"
NUM_EPOCHS = 10

def save_current_model(n_epoch):
	save_path = Path(f'./models/{EXPERIMENT_NAME}_epoch_{n_epoch}.pt')
	save_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), save_path)
	print(f"Model saved to {save_path}")

save_current_model(0)

# === Tracking variables ===
train_losses = []
train_accs = []
test_accs = []

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for inputs, targets in tqdm(train_loader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)

		if task == 'multi-label, binary-class':
			targets = targets.to(torch.float32)
			loss = criterion(outputs, targets)
		else:
			targets = targets.squeeze().long()
			loss = criterion(outputs, targets)
			_, predicted = outputs.max(1)
			correct += predicted.eq(targets).sum().item()
			total += targets.size(0)

		loss.backward()
		optimizer.step()
		running_loss += loss.item()

	# Store training metrics
	train_losses.append(running_loss / len(train_loader))
	if task != 'multi-label, binary-class':
		train_acc = correct / total
		train_accs.append(train_acc)
	else:
		train_metrics = test('train')  # for multi-label case, get acc from test function
		train_accs.append(train_metrics[1])  # assuming acc is at index 1

	# Evaluate on test set
	test_metrics = test('test')
	print(test_metrics)
	test_accs.append(test_metrics[1])  # assuming acc is at index 1

	print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
		  f"Loss: {train_losses[-1]:.4f} "
		  f"Train Acc: {train_accs[-1]:.4f} "
		  f"Test Acc: {test_accs[-1]:.4f}")
	
	# === Save model ===
	save_current_model(epoch+1)

# === Plotting ===
epochs = range(1, NUM_EPOCHS + 1)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot training loss on left y-axis
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
l1, = ax1.plot(epochs, train_losses, color='tab:red', label='Train Loss')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a second y-axis for accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:blue')
l2, = ax2.plot(epochs, train_accs, color='tab:blue', linestyle='--', label='Train Acc')
l3, = ax2.plot(epochs, test_accs, color='tab:green', linestyle='-', label='Test Acc')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Combine legends
lines = [l1, l2, l3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Training Progress')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# **RISE**

# %%
# RISE implementation
from skimage.transform import resize
import torch.nn.functional as F

class PyTorchModelWrapper:
    def __init__(self, model, input_size, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.input_size = input_size

    def run_on_batch(self, x):
        with torch.no_grad():
            x = torch.tensor(x.transpose(0, 3, 1, 2)).float().to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs
    
def generate_masks(N, s, p1):
    """
    Generate random masks for the RISE algorithm.

    Parameters:
    ----------
    N : int
        Number of masks to generate (i.e., how many masked samples will be used 
        to estimate the saliency map). A higher value leads to better estimation 
        but increases computational cost.

    s : int
        Spatial resolution of the small binary grid (s x s) before upsampling.
        Controls how coarse or fine the masks are before interpolation.
        Typical values are like 7 or 14.

    p1 : float
        Probability that each cell in the small grid is set to 1 (i.e., 
        the region is kept instead of masked). Controls the sparsity of the mask.
        Typical values not specified in the paper but is 0.5 in the RISE
        Github implementation

    Returns:
    -------
    masks : np.ndarray
        An array of shape (N, H, W, 1) containing the upsampled and randomly 
        shifted masks, where H and W match the model's input size.
    """
    cell_size = np.ceil(np.array(model.input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *model.input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + model.input_size[0], y:y + model.input_size[1]]
    masks = masks.reshape(-1, *model.input_size, 1)
    return masks

batch_size = 100

def explain(model, inp, masks, N, p1):
    preds = []
    # Make sure multiplication is being done for correct axes
    masked = inp * masks
    for i in tqdm(range(0, N, batch_size), desc='Explaining'):
        preds.append(model.run_on_batch(masked[i:min(i+batch_size, N)]))
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *model.input_size)
    sal = sal / N / p1
    return sal

# %% [markdown]
# Experiment on different epochs. But didn't find much diff.

# %%
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

# Constants
MODEL_PATH_PATTERN = "./models/RISE_224x224_RESTNET18_epoch_{}.pt"
EPOCHS = range(1, 11)  # Adjust as needed
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load sample image (e.g. from PathMNIST, resized to 224x224)
transform_resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_tensor, label = test_dataset[0]
img = img_tensor.numpy().transpose(1, 2, 0)
img = img * 0.5 + 0.5 
img = (img * 255).astype(np.uint8)  # Convert to uint8 [0,255]
# plt.imshow(img)
plt.imsave("./results/origin.png", img)
# RISE Parameters
N = 2000
s = 8
p1 = 0.5
input_size = (224, 224)

# Generate masks once for all runs
model = type('Temp', (), {})()  # dummy object to pass input size to mask gen
model.input_size = input_size
masks = generate_masks(N, s, p1)

# Explain with each saved model
for epoch in EPOCHS:
    model_path = MODEL_PATH_PATTERN.format(epoch)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue

    # Load model
    net = resnet18(num_classes=9)  # PathMNIST has 9 classes
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    wrapped_model = PyTorchModelWrapper(net, input_size = (224, 224),device=device)

    # Generate saliency map
    sal = explain(wrapped_model, x, masks, N, p1)
    class_idx = int(label)

    # Plot and save
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(img)
    plt.imshow(sal[class_idx], cmap='jet', alpha=0.5)
    plt.title(f"Epoch {epoch} - Class {class_idx}")

    save_path = os.path.join(RESULTS_DIR, f"saliency_epoch_{epoch}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Normalize saliency map to [0, 1]
    sal_map = sal[class_idx]
    sal_map -= sal_map.min()
    sal_map /= sal_map.max() + 1e-8

    # Convert original image and saliency to PIL
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    origin_path = os.path.join(RESULTS_DIR, "origin.png")
    if not os.path.exists(origin_path):
        img_pil.save(origin_path)
        print(f"Saved: origin.png")
    sal_color = plt.cm.jet(sal_map)[:, :, :3]  # remove alpha channel
    sal_overlay = (sal_color * 255).astype(np.uint8)
    sal_pil = Image.fromarray(sal_overlay)

    # Blend saliency with original image using 0.5 alpha
    blended = Image.blend(img_pil.convert('RGB'), sal_pil.convert('RGB'), alpha=0.3)

    # Save blended image
    blended.save(os.path.join(RESULTS_DIR, f"rise_epoch_{epoch}.png"))
    print(f"Saved: rise_epoch_{epoch}.png")




# %% [markdown]
# ## Experiment on xai_data_224 for comparison

# %%
# RISE implementation
from skimage.transform import resize
import torch.nn.functional as F

class PyTorchModelWrapper:
    def __init__(self, model, input_size, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.input_size = input_size

    def run_on_batch(self, x):
        with torch.no_grad():
            x = torch.tensor(x.transpose(0, 3, 1, 2)).float().to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        # print(probs)
        return probs
    
def generate_masks(N, s, p1):
    """
    Generate random masks for the RISE algorithm.

    Parameters:
    ----------
    N : int
        Number of masks to generate (i.e., how many masked samples will be used 
        to estimate the saliency map). A higher value leads to better estimation 
        but increases computational cost.

    s : int
        Spatial resolution of the small binary grid (s x s) before upsampling.
        Controls how coarse or fine the masks are before interpolation.
        The larget the s the smaller the grid size.
        Typical values are like 7 or 14.

    p1 : float
        Probability that each cell in the small grid is set to 1 (i.e., 
        the region is kept instead of masked). Controls the sparsity of the mask.
        Typical values not specified in the paper but is 0.5 in the RISE
        Github implementation

    Returns:
    -------
    masks : np.ndarray
        An array of shape (N, H, W, 1) containing the upsampled and randomly 
        shifted masks, where H and W match the model's input size.
    """
    cell_size = np.ceil(np.array(model.input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *model.input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + model.input_size[0], y:y + model.input_size[1]]
    masks = masks.reshape(-1, *model.input_size, 1)
    return masks

batch_size = 100

def explain(model, inp, masks, N, p1):
    preds = []
    # Make sure multiplication is being done for correct axes
    masked = inp * masks
    for i in tqdm(range(0, N, batch_size), desc='Explaining'):
        preds.append(model.run_on_batch(masked[i:min(i+batch_size, N)]))
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *model.input_size)
    sal = sal / N / p1
    return sal

# %%
## Dataloader to load the saved tensor instead for explainability experiments.
## Usage: 
# from torch.utils.data import DataLoader

# xai_dataset = SavedTensorDataset('./xai_data_28x28') # or xai_data_224 for the resnet experiment
# xai_loader = DataLoader(xai_dataset, batch_size=1, shuffle=False)

# for img_tensor, label in xai_loader:
#     # Use `img_tensor` for inference, saliency, etc.
#     print(img_tensor.shape, label.item())

import os
import torch
from torch.utils.data import Dataset

class SavedTensorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted([
            f for f in os.listdir(root_dir) if f.endswith(".pt")
        ])
        self.labels = [int(f.split('_')[1]) for f in self.file_list]  # from filename

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        tensor = torch.load(file_path)
        label = self.labels[idx]
        return tensor, label
    


# %%
from torch.utils.data import DataLoader

xai_dataset = SavedTensorDataset('./xai_data_224') # or xai_data_224 for the resnet experiment
xai_loader = DataLoader(xai_dataset, batch_size=1, shuffle=False)

for img_tensor, label in xai_loader:
    # Use `img_tensor` for inference, saliency, etc.
    print(img_tensor.shape, label.item())


import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

# Constants
EXPERIMENT_NAME = "RISE_ON_RESNET_XAIDATASET_N8000_S20"
MODEL_PATH = "./models/RISE_224x224_RESTNET18_epoch_10.pt"
RESULTS_DIR = F"./results/{EXPERIMENT_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# RISE Parameters
N = 8000
s = 20
p1 = 0.5
input_size = (224, 224)

# Generate masks once for all runs
model = type('Temp', (), {})()  # dummy object to pass input size to mask gen
model.input_size = input_size
masks = generate_masks(N, s, p1)

# Explain
model_path = MODEL_PATH
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")

# Load model
net = resnet18(num_classes=9)  # PathMNIST has 9 classes
net.load_state_dict(torch.load(model_path, map_location='cpu'))
net.eval()
wrapped_model = PyTorchModelWrapper(net, input_size = (224, 224),device=device)

for idx, (img_tensor, label) in enumerate(xai_loader):
    img_tensor = img_tensor.squeeze(0)  # [3, 224, 224]
    label = int(label.item())

    # Reconstruct file name using known convention
    fname = f"class_{label}_{idx % 2}"  # Assuming 2 samples per class in order

    # Unnormalize for visualization
    vis_img = img_tensor * 0.5 + 0.5
    vis_img = vis_img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    vis_img_pil = Image.fromarray(vis_img)

    # RISE input: HWC, float32, [0,1]
    x = (vis_img / 255.0).astype(np.float32)

    # Generate saliency map
    sal = explain(wrapped_model, x, masks, N, p1)
    sal_map = sal[label]

    # Normalize saliency
    sal_map -= sal_map.min()
    sal_map /= sal_map.max() + 1e-8
    sal_color = plt.cm.jet(sal_map)[:, :, :3]
    sal_overlay = (sal_color * 255).astype(np.uint8)
    sal_pil = Image.fromarray(sal_overlay)

    # Blend with alpha
    blended = Image.blend(vis_img_pil.convert('RGB'), sal_pil.convert('RGB'), alpha=0.5)

    # Save image
    blended.save(os.path.join(RESULTS_DIR, f"rise_{fname}.png"))
    print(f"Saved: rise_{fname}.png")

# %%
### Currently the project scope ends here. The following is from the original sample and remains unchanged.


