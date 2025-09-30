# read .tif files from a given folder and create a dataset for pytorch

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class PIVDataset(Dataset):
    """0.tif is the first snap in snap1 folder. These are the inputs to the model,
    0.tif is the second snap with PDH information in particle folder. These are the outputs to the model,
    0.tif is the second snap with truth data in fluid folder. These are the truth data for the model."""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.snap1_dir = os.path.join(root_dir, 'snap1')
        self.particle_dir = os.path.join(root_dir, 'particle')
        self.fluid_dir = os.path.join(root_dir, 'fluid')
        if os.path.exists(os.path.join(root_dir, 'scalars.csv')):
            self.scalars = pd.read_csv(os.path.join(root_dir, 'scalars.csv'))
        else:
            self.scalars = None
            print('No scalars.csv file found in the root directory. Scalars will not be available for this dataset.')

        self.snap1_images = sorted([i for i in os.listdir(self.snap1_dir) if i.endswith('.tif')])
        self.particle_images = sorted([i for i in os.listdir(self.particle_dir) if i.endswith('.tif')])
        self.fluid_images = sorted([i for i in os.listdir(self.fluid_dir) if i.endswith('.tif')])

    def __len__(self):
        return len(self.snap1_images)

    def __getitem__(self, idx):
        snap1_img_path = os.path.join(self.snap1_dir, self.snap1_images[idx])
        particle_img_path = os.path.join(self.particle_dir, self.particle_images[idx])
        fluid_img_path = os.path.join(self.fluid_dir, self.fluid_images[idx])

        # open images using PIL
        snap1_image = Image.open(snap1_img_path)
        particle_image = Image.open(particle_img_path)
        fluid_image = Image.open(fluid_img_path)

        # get scalars for this image from the csv file
        if self.scalars is not None:
            scalars = np.array([self.scalars['mach'][0], self.scalars['reynolds number'][0]])
        else:
            scalars = float('nan')

        # apply transform to convert images to tensors in float64 and keep only 3 channels
        transform_to_tensor = transforms.ToTensor()
        snap1_image = transform_to_tensor(snap1_image)[:3]
        particle_image = transform_to_tensor(particle_image)[:3]
        fluid_image = transform_to_tensor(fluid_image)[:3]

        sample = {'snap1': snap1_image, 'particle': particle_image, 'fluid': fluid_image, 'scalars': scalars}

        if self.transform:
            sample = self.transform(sample)

        return sample

