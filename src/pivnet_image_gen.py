# File to generate images from the BICSNet model

import torch
from bicsnet import Net
from loader import PIVDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
# Strip the "module." prefix from all keys -- to run on CPU
from collections import OrderedDict
sns.set_theme('paper')

# set the default data type to float64
torch.set_default_dtype(torch.float64)

def load_data(dataset_dir: str, batch_size: int):
    """
    Load all images from the dataset directory into a single DataLoader without splitting.
    """
    dataset = PIVDataset(root_dir=dataset_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def save_image(image, directory: str, name: str):
    """
    Save the image to the directory with the given name
    :param image:
    :param directory:
    :param name:
    :return:
    """
    # check if the directory exists and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    # check if name.tif exists and remove it
    if os.path.exists(os.path.join(directory, name + '.tif')):
        os.remove(os.path.join(directory, name + '.tif'))

    # save the image
    plt.imsave(os.path.join(directory, name + '.tiff'), image)

    return


def rename_files(directory: str):
    """
    rename all the files with .tiff extension to .tif
    :param directory:
    :return:
    """
    for filename in os.listdir(directory):
        if filename.endswith('.tiff'):
            os.rename(os.path.join(directory, filename), os.path.join(directory, filename[:-1]))
    return


def generate_images(model, test_loader, device, save_dir1: str, save_dir2: str):
    count = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc='    Generating images from model'):
            inputs1, inputs2 = data['snap1'].to(device), data['particle'].to(device)
            labels1, labels2 = data['snap1'].to(device), data['fluid'].to(device)
            scalars = data['scalars'].to(device)
            outputs1, outputs2 = model(inputs1, inputs2, scalars)
            # save the model outputs as two PIV snaps in the save_dir with the same name as the input snaps
            for i in tqdm(range(len(outputs1)), desc='        Saving images'):
                # get the snaps in 3x256x256 format
                snap1 = outputs1[i, ...].cpu().numpy()
                snap2 = outputs2[i, ...].cpu().numpy()
                input_snap1 = inputs1[i, ...].cpu().numpy()
                input_snap2 = inputs2[i, ...].cpu().numpy()
                label_snap1 = labels1[i, ...].cpu().numpy()
                label_snap2 = labels2[i, ...].cpu().numpy()
                # normalize to 0 to 255
                snap1 = (snap1 - snap1.min()) / (snap1.max() - snap1.min())
                snap2 = (snap2 - snap2.min()) / (snap2.max() - snap2.min())
                input_snap1 = (input_snap1 - input_snap1.min()) / (input_snap1.max() - input_snap1.min())
                input_snap2 = (input_snap2 - input_snap2.min()) / (input_snap2.max() - input_snap2.min())
                label_snap1 = (label_snap1 - label_snap1.min()) / (label_snap1.max() - label_snap1.min())
                label_snap2 = (label_snap2 - label_snap2.min()) / (label_snap2.max() - label_snap2.min())
                # save the output snaps in .tif format
                save_image(snap1.transpose(1, 2, 0), save_dir1, str(count))
                save_image(snap2.transpose(1, 2, 0), save_dir2, str(count))
                save_image(input_snap1.transpose(1, 2, 0), save_dir1, str(count) + '_input')
                save_image(input_snap2.transpose(1, 2, 0), save_dir2, str(count) + '_input')
                save_image(label_snap1.transpose(1, 2, 0), save_dir1, str(count) + '_truth')
                save_image(label_snap2.transpose(1, 2, 0), save_dir2, str(count) + '_truth')
                # rename to .tif
                rename_files(save_dir1)
                rename_files(save_dir2)
                count += 1

    return


if __name__ == "__main__":
    base_path = './data/test_images/'
    # remove associated model output folders
    folders = ['model_outputs1', 'model_outputs2']
    # check if the folders exist and create them if they don't
    Path(base_path + folders[0]).mkdir(parents=True, exist_ok=True)
    Path(base_path + folders[1]).mkdir(parents=True, exist_ok=True)
    for folder in folders:
        path = os.path.join(base_path, folder)
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    # Select best available device adaptively: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif (
        hasattr(torch.backends, 'mps')
        and torch.backends.mps.is_available()
        and platform.machine().lower() == 'arm64'
    ):
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model_dir = './checkpoints/'
    save_dir1 = base_path + folders[0]
    save_dir2 = base_path + folders[1]

    # load the model
    num_layers_encoder, num_layers_decoder = 5, 5  # Number of layers in encoder and decoder
    use_scalars = True  # Use scalar embeddings
    scalars_input_dim, scalar_embed_dim = 2, 128  # Scalar input dimension and embedding dimension
    model = Net(num_layers_encoder=num_layers_encoder, num_layers_decoder=num_layers_decoder, use_scalars=use_scalars,
                scalar_input_dim=scalars_input_dim, scalar_embed_dim=scalar_embed_dim)
    # Wrap with DataParallel only if multiple CUDA GPUs are available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load checkpoint to the selected device (handles CPU/MPS/CUDA)
    state_dict = torch.load(model_dir + 'best_model.pth', map_location=device)

    # Load the state dict robustly regardless of DataParallel prefixing
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        # Handle mismatch of 'module.' prefix
        keys = list(state_dict.keys())
        if len(keys) > 0 and keys[0].startswith('module.'):
            # Strip 'module.' if model is not DataParallel
            if not isinstance(model, torch.nn.DataParallel):
                from collections import OrderedDict
                new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
                model.load_state_dict(new_state_dict, strict=False)
            else:
                raise
        else:
            # Add 'module.' if model is DataParallel
            if isinstance(model, torch.nn.DataParallel):
                from collections import OrderedDict
                new_state_dict = OrderedDict((f'module.{k}', v) for k, v in state_dict.items())
                model.load_state_dict(new_state_dict, strict=False)
            else:
                raise
    model.to(device)

    # load all data without split
    test_loader = load_data(base_path, 8)

    # generate the images
    generate_images(model, test_loader, device, save_dir1, save_dir2)

    print('Images generated successfully')