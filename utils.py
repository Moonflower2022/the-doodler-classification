import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from quickdraw import QuickDrawDataGroup
import random
from PIL import Image
import matplotlib.pyplot as plt

class QuickDrawDataset(Dataset):
    def __init__(self, drawings, drawing_count, label):
        self.drawings = drawings
        self.drawing_count = drawing_count
        self.label = label

    def __len__(self):
        return self.drawing_count

    def __getitem__(self, idx):
        if idx >= self.drawing_count:
            raise IndexError("Index out of bounds")
        
        # Get the raw drawing
        drawing = self.drawings[idx]
        
        # Assuming drawing.get_image() returns a PIL image or something similar
        img = drawing.get_image()

        # Convert the drawing to an image (28x28 pixels)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img).resize((28, 28)).convert('L')
        else:
            img = img.resize((28, 28)).convert('L')
        
        # Convert the image to a numpy array and normalize the pixel values
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to a PyTorch tensor
        img_tensor = torch.tensor(img).unsqueeze(0)  # add channel dimension
        # Return the image tensor and the label
        return img_tensor, self.label
    
def load_quickdraw_data(classes_file_name, num_drawings, train_test_ratio):
    with open(f"categories/{classes_file_name}", "r") as file:
        categories = file.read().split("\n")

    num_classes = len(categories)

    raw = [QuickDrawDataGroup(category, max_drawings=num_drawings) for category in categories]

    train_datasets = []
    test_datasets = []

    for i, data_group in enumerate(raw):
        drawings = list(data_group.drawings)
        random.shuffle(drawings)

        cutoff = int(train_test_ratio * len(drawings))

        train_datasets.append(QuickDrawDataset(drawings[:cutoff], cutoff, label=i))
        test_datasets.append(QuickDrawDataset(drawings[cutoff:], len(drawings) - cutoff, label=i))

    combined_dataset_train = ConcatDataset(train_datasets)
    combined_dataset_test = ConcatDataset(test_datasets)

    return num_classes, categories, combined_dataset_train, combined_dataset_test

def show_image(image_tensor):
    img_array = image_tensor.squeeze(0).numpy()

    # Rescale pixel values back to [0, 255]
    img_array = (img_array * 255).astype(np.uint8)

    # Convert to PIL Image
    img_pil = Image.fromarray(img_array)

    # Display the image using PIL or matplotlib
    img_pil.show()  # Using PIL's show method
    # Or using matplotlib
    plt.imshow(img_array, cmap='gray')
    plt.show()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)