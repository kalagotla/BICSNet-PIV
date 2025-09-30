# File to generate images from the PIVNet model

import torch
from bicsnet import Net
from loader import PIVDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# Strip the "module." prefix from all keys -- to run on CPU
from collections import OrderedDict
sns.set_theme('paper')

# set the default data type to float64
torch.set_default_dtype(torch.float64)

def load_data(dataset_dir: str, test_size: float, random_state: int, batch_size: int):
    dataset = PIVDataset(root_dir=dataset_dir)  # Load the dataset
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=random_state)
    train_indices, val_indices = train_test_split(train_indices, test_size=test_size, random_state=random_state)

    # Create subsets and data loaders for each split
    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader


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
    for i in range(768, 2304+128, 128):
        base_path = '../data/FSU_first_shock/' + str(i) + '/'
        # remove associated model output folders
        folders = ['model_outputs1', 'model_outputs2']
        # check if the folders exist and create them if they don't
        Path(base_path + folders[0]).mkdir(parents=True, exist_ok=True)
        Path(base_path + folders[1]).mkdir(parents=True, exist_ok=True)
        for folder in folders:
            path = os.path.join(base_path, folder)
            for filename in os.listdir(path):
                os.remove(os.path.join(path, filename))
        # use cpu for visualization
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model_dir = '../data/bicsnet/'
        save_dir1 = base_path + folders[0]
        save_dir2 = base_path + folders[1]

        # load the model
        num_layers_encoder, num_layers_decoder = 5, 5  # Number of layers in encoder and decoder
        use_scalars = True  # Use scalar embeddings
        scalars_input_dim, scalar_embed_dim = 2, 128  # Scalar input dimension and embedding dimension
        model = Net(num_layers_encoder=num_layers_encoder, num_layers_decoder=num_layers_decoder, use_scalars=use_scalars,
                    scalar_input_dim=scalars_input_dim, scalar_embed_dim=scalar_embed_dim)
        # use the line below to load model from multiple GPUs
        model = torch.nn.DataParallel(model)
        # Load the original state_dict
        state_dict = torch.load(model_dir + 'best_model.pth')

        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     new_key = k.replace("module.", "")  # Remove "module." prefix
        #     new_state_dict[new_key] = v
        # state_dict = new_state_dict.copy()

        # Load the modified state_dict into the model
        model.load_state_dict(state_dict)
        model.to(device)

        # load the data
        test_loader = load_data(base_path, 0.90, 43, 8)[1]

        # generate the images
        generate_images(model, test_loader, device, save_dir1, save_dir2)

        # temp code to flip the images
        # folders = ['snap1', 'fluid', 'particle']
        # for folder in folders:
        #     path = os.path.join(base_path, folder)
        #     for i in tqdm(range(10000), desc='Loading images from ' + folder):
        #         image = plt.imread(os.path.join(path, str(i) + '.tif'))
        #         # flip the image
        #         image = np.flipud(np.fliplr(image)).copy()
        #         # save the image back
        #         plt.imsave(os.path.join(path, str(i) + '.tiff'), image)
        #         os.remove(os.path.join(path, str(i) + '.tif'))
        #         # rename to .tif
        #         rename_files(path)