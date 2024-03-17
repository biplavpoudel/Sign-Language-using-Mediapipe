import torch
import numpy as np
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
plt.ion()


def device_check():
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device for training the CNN model.")
    return device


def create_dataset():

    # Preprocessing-function
    train_transforms = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color jitter
        # transforms.RandomRotation(degrees=3),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        # Add random affine transformation
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # Add random perspective transformation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([-0.209, -0.174,  0.009], [1.230, 1.304, 1.335])
    ])
    # inference_transforms = transforms.Compose([
    #     transforms.Resize(size=232),
    #     transforms.CenterCrop(size=224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([-0.209, -0.174, 0.009], [1.230, 1.304, 1.335])
    ])

    train_dataset = ImageFolder(root="./Input/landmark_images/", transform=train_transforms)

    class_names = train_dataset.classes
    print(f"\nThe class names are:\n {class_names}\n")

    return train_dataset, class_names


def split_dataset(train_ds, device):

    # Percentage of validation split
    valid_ratio = 0.2

    valid_size = int(len(train_ds) * valid_ratio)
    print(f"Validation set size: {valid_size}")
    train_size = len(train_ds) - valid_size
    print(f"Train set size: {train_size}")

    train_dataset, valid_dataset = random_split(train_ds, [train_size, valid_size])

    print(f"Train dataset has: {len(train_ds)} images, which are split into:\n"
          f" {train_size} train samples and\n"
          f" {valid_size} validation samples.")

    # wrap datasets into iterable datasets using DataLoader

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4)

    loaders = dict(train=train_loader, valid=valid_loader)
    return loaders


# def calculate_img_stats(dataset):
#     print("Calculating mean and std of images....")
#     # Initialize lists to store pixel values
#     num_batches = 300
#     pixel_values = []
#     # lets use cpu instead of gpu due to constant out of cuda memory issues on larger batches
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=False)
#     # Iterate over a subset of batches to collect pixel values
#     for i, (images, _) in enumerate(dataloader):
#         if i >= num_batches:
#             break
#         pixel_values.append(images.numpy())
#
#     # Concatenate pixel values along the batch dimension
#     pixel_values = np.concatenate(pixel_values, axis=0)
#
#     # Calculate mean and standard deviation across color channels
#     mean = np.mean(pixel_values, axis=(0, 2, 3))
#     std = np.std(pixel_values, axis=(0, 2, 3))
#
#     print("Mean value of the image is:", mean)
#     print("Std value of the image is:", std)
#
#     return mean, std


# Visualize datasets
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))  # (1 is height, 2 is width, 0 is batch size)
    mean = np.array([0.485, 0.456, 0.406])
    # mean = np.array([-0.209, -0.174,  0.009])
    std = np.array([0.229, 0.224, 0.225])
    # std = np.array([1.230, 1.304, 1.335])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    device_name = device_check()
    train, classnames = create_dataset()
    # mean, std = calculate_img_stats(train)
    dictionary = split_dataset(train, device_name)
    # Get a batch of training data
    image, classes = next(iter(dictionary['train']))
    image = image[:8]
    classes = classes[:8]

    # Make a grid from batch
    out = torchvision.utils.make_grid(image, nrow=4)
    imshow(out, title=[classnames[x] for x in classes])
