import torch
import numpy as np
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import mediapipe as mp
plt.ion()


def device_check():
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device for training the CNN model.")
    return device


if __name__ == "__main__":
    device = device_check()