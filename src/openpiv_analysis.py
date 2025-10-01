"""
Utilities for running OpenPIV analysis on BICSNet outputs, raw PIV inputs, and ground truth.

This module encapsulates the OpenPIV settings and exposes simple functions:
  - openpiv_model(i)
  - openpiv_piv(i)
  - openpiv_truth(i)

Each function runs a 3-pass window deformation PIV pipeline and writes
results into a corresponding folder: `./_model_multipass`, `./_piv_multipass`, `./_truth_multipass`.
"""

from __future__ import annotations

from openpiv import windef


def _base_settings() -> "windef.PIVSettings":
    settings = windef.PIVSettings

    # Region of interest
    settings.roi = 'full'

    # Image preprocessing
    settings.dynamic_masking_method = 'None'
    settings.dynamic_masking_threshold = 0.005
    settings.dynamic_masking_filter_size = 7
    settings.deformation_method = 'symmetric'

    # Processing Parameters
    settings.correlation_method = 'circular'
    settings.normalized_correlation = False
    settings.num_iterations = 3
    settings.windowsizes = (64, 32, 16)
    settings.overlap = (32, 16, 8)
    settings.subpixel_method = 'gaussian'
    settings.interpolation_order = 3
    settings.scaling_factor = 34133.33  # pixel/meter
    settings.dt = 1e-6                # seconds

    # Signal to noise ratio options
    settings.sig2noise_method = 'peak2peak'
    settings.sig2noise_mask = 2

    # Vector validation options
    settings.validation_first_pass = True
    settings.min_max_u_disp = (-30, 30)
    settings.min_max_v_disp = (-30, 30)
    settings.std_threshold = 7
    settings.median_threshold = 3
    settings.median_size = 1
    settings.sig2noise_threshold = 1.2

    # Outlier replacement and smoothing
    settings.replace_vectors = True
    settings.smoothn = True
    settings.smoothn_p = 0.5
    settings.filter_method = 'localmean'
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2

    # Output options
    settings.save_plot = False
    settings.show_plot = False
    settings.scale_plot = 1e5

    return settings


def openpiv_model(i: int) -> None:
    """Run OpenPIV analysis on BICSNet model outputs for sample i."""
    settings = _base_settings()
    settings.filepath_images = './'
    settings.save_path = './data/test_images/openpiv_analysis/model'
    settings.save_folder_suffix = '/' + str(i)
    settings.frame_pattern_a = './data/test_images/model_outputs1/' + str(i) + '.tif'
    settings.frame_pattern_b = './data/test_images/model_outputs2/' + str(i) + '.tif'
    windef.piv(settings)


def openpiv_piv(i: int) -> None:
    """Run OpenPIV analysis on raw PIV inputs for sample i."""
    settings = _base_settings()
    settings.filepath_images = './'
    settings.save_path = './data/test_images/openpiv_analysis/piv'
    settings.save_folder_suffix = '/' + str(i)
    settings.frame_pattern_a = './data/test_images/model_outputs1/' + str(i) + '_input.tif'
    settings.frame_pattern_b = './data/test_images/model_outputs2/' + str(i) + '_input.tif'
    windef.piv(settings)


def openpiv_truth(i: int) -> None:
    """Run OpenPIV analysis on ground truth for sample i."""
    settings = _base_settings()
    settings.filepath_images = './'
    settings.save_path = './data/test_images/openpiv_analysis/truth'
    settings.save_folder_suffix = '/' + str(i)
    settings.frame_pattern_a = './data/test_images/model_outputs1/' + str(i) + '_truth.tif'
    settings.frame_pattern_b = './data/test_images/model_outputs2/' + str(i) + '_truth.tif'
    windef.piv(settings)


__all__ = [
    'openpiv_model',
    'openpiv_piv',
    'openpiv_truth',
]
