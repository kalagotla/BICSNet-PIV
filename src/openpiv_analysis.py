from openpiv import tools, pyprocess, validation, filters, scaling

import shutil
import glob
import os
from pathlib import Path
from tqdm import tqdm
from mpi4py import MPI
from imageio import v3 as imageio
import numpy as np
import pandas as pd
import argparse


# function to run the PIV analysis
def openpiv_default_run(im1, im2, window_size, overlap, search_size, dt=1e-6, savefile='test.txt'):
    """ default settings for OpenPIV analysis using
    extended_search_area_piv algorithm for two images

    Inputs:
        im1,im2 : str,str = path of two image
    """
    frame_a = tools.imread(im1)
    frame_b = tools.imread(im2)

    u, v, sig2noise = pyprocess.extended_search_area_piv(frame_a, frame_b,
                                                         window_size=window_size,
                                                         overlap=overlap,
                                                         dt=dt,
                                                         search_area_size=search_size,
                                                         sig2noise_method='peak2mean',
                                                         correlation_method='circular',
                                                         normalized_correlation=True)
    # plt.hist(sig2noise.flatten())
    x, y = pyprocess.get_rect_coordinates(frame_a.shape,
                                          search_size,
                                          overlap)
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=105.)

    x, y, u, v = tools.transform_coordinates(x, y, u, v)

    tools.save(savefile, x, y, u, v)

    return


def find_scalars_files(root_dir):
    # Recursively search for scalars.csv files in the root_dir
    pattern = os.path.join(root_dir, '**', 'scalars.csv')
    # return the directories containing the scalars.csv files and end with a '/'
    return [os.path.dirname(path) + '/' for path in glob.glob(pattern, recursive=True)]

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # analysis parameters
    search_size = 32
    overlap = 24
    window_size = 32
    root_dir = './data/test_images/'
    redo_analysis = True
    if rank == 0:
        base_paths = find_scalars_files(root_dir)
    else:
        base_paths = None
    base_paths = comm.bcast(base_paths, root=0)
    base_paths = base_paths[rank::size]
    for base_path in tqdm(base_paths, desc='Iterating over directories'):
        # if os.path.exists(base_path + f'piv_outputs_ws_{window_size}/truth_127.txt') and not redo_analysis:
        if os.path.exists(base_path + 'intensity_metrics.csv') and not redo_analysis:
            print(f'Skipping {base_path} as PIV processing is already done.')
        else:
            print(f'Running cross-correlation for {base_path}...')
            # paths for the images
            path_snap1 = base_path + 'model_outputs1/'
            path_snap2 = base_path + 'model_outputs2/'
            save_path = base_path + f'piv_outputs_ws_{window_size}/'
            Path(save_path).mkdir(parents=True, exist_ok=True)

            # get all the files as a list in the directory for model, input and truth
            model1, model2, input1, input2, truth1, truth2 = [], [], [], [], [], []
            count = 0
            for filename in os.listdir(path_snap1):
                if filename.endswith('.tif'):
                    count += 1

            count = int(count / 3)

            for i in range(count):
                model1.append(path_snap1 + str(i) + '.tif')
                model2.append(path_snap2 + str(i) + '.tif')
                input1.append(path_snap1 + str(i) + '_input.tif')
                input2.append(path_snap2 + str(i) + '_input.tif')
                truth1.append(path_snap1 + str(i) + '_truth.tif')
                truth2.append(path_snap2 + str(i) + '_truth.tif')

            # run the PIV analysis
            for i in tqdm(range(count), desc='Running PIV analysis'):
                openpiv_default_run(model1[i], model2[i], search_size, overlap, window_size, dt=1e-6,
                                    savefile=f'{save_path}model_{i}.txt')
                openpiv_default_run(input1[i], input2[i], search_size, overlap, window_size, dt=1e-6,
                                    savefile=f'{save_path}input_{i}.txt')
                openpiv_default_run(truth1[i], truth2[i], search_size, overlap, window_size, dt=1e-6,
                                    savefile=f'{save_path}truth_{i}.txt')

            # do the intensity analysis
            model_mse, input_mse = [], []
            model_psnr, input_psnr = [], []
            for i in tqdm(range(count), desc='Running intensity analysis'):
                # compute image pixel MSE
                model_im1 = imageio.imread(model1[i])
                model_im2 = imageio.imread(model2[i])
                input_im1 = imageio.imread(input1[i])
                input_im2 = imageio.imread(input2[i])
                truth_im1 = imageio.imread(truth1[i])
                truth_im2 = imageio.imread(truth2[i])
                mse_model1 = np.mean((model_im1 - truth_im1) ** 2)
                mse_input1 = np.mean((input_im1 - truth_im1) ** 2)
                mse_model2 = np.mean((model_im2 - truth_im2) ** 2)
                mse_input2 = np.mean((input_im2 - truth_im2) ** 2)
                # psnr_model1 = 10 * np.log10(255**2 / mse_model1)
                # psnr_input1 = 10 * np.log10(255**2 / mse_input1)
                psnr_model2 = 10 * np.log10(255**2 / mse_model2)
                psnr_input2 = 10 * np.log10(255**2 / mse_input2)
                # average
                model_mse.append((mse_model1 + mse_model2) / 2)
                input_mse.append((mse_input1 + mse_input2) / 2)
                model_psnr.append(psnr_model2)
                input_psnr.append(psnr_input2)

            # save the intensity analysis
            df = pd.DataFrame({'model_mse': model_mse, 'input_mse': input_mse, 'model_psnr': model_psnr, 'input_psnr': input_psnr})
            df.to_csv(f'{base_path}intensity_metrics.csv', index=False)
    comm.Barrier()


