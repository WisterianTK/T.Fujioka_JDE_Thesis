import numpy as np
from scipy.stats import zscore
from matplotlib import pyplot as plt
from scipy.io import savemat
import os

import fusilib.config
# # Enter the path to the downloaded "Subjects" directory.
# # By default, the path is set to the current working directory.
data_location = r'C:\Users\TK\Documents\SC\Tatsuki_Thesis\Nunez_repo\fusi\Subjects'

# Where you want to save the extracted data
save_location = r'C:\Users\TK\Documents\SC\Tatsuki_Thesis\Nunez_repo\fusi\Subjects_for_MATLAB'
os.makedirs(save_location, exist_ok=True)

fusilib.config.set_dataset_path(data_location)

from fusilib import handler2 as handler
from fusilib import metahelper

EXPERIMENTS = [('CR017', '2019-11-13'),
               ('CR017', '2019-11-14'),
               #('CR019', '2019-11-26'), This one doesn't have neuro pixel
               # ('CR019', '2019-11-27'), These have only one probe
               # ('CR020', '2019-11-20'),
               # ('CR020', '2019-11-21'),
               # ('CR020', '2019-11-22'),
               ('CR022', '2020-10-07'),
               # ('CR022', '2020-10-11'),  13 block has no neuro pixel data
               # ('CR024', '2020-10-29'),  7 block has no neuro pixel data
               ]

stimulus_label = {'left': 0,
                  'center': 1,
                  'blank': 2,
                  'right': 3
                  }

probe_name_rh = 'probe00'
probe_name_lh = 'probe01'

fusi_dt = 0.300  # fUSI sampling rate in [sec]
mua_probe_widthum = 200  # width of probe

## Iterate over all subjects
for idx, (subjectID, date) in enumerate(EXPERIMENTS):
    subject = handler.MetaSession(subject_name=subjectID, session_name=date, verbose=False)
    spontaneous_blocks = subject.experiment_get_task_blocks('spontaneous')
    checkerboard_blocks = subject.experiment_get_task_blocks('checkerboard')

    ## For spontaneous
    for block_id_sp in spontaneous_blocks:
        subject_block = handler.MetaBlock(subjectID, date, block_id_sp)

        ## Code to extract fUS
        fusi_times, fusi_data = subject_block.fusi_get_data(dt_ms=300, window_ms=400, svddrop=15,
                                    freq_cutoffhz=15, roi_name=None, mirrored=False, verbose=True)

        ## Code to extract spatial resolution and depth width information (in mm)
        width_resolution_mm, depth_resolution_mm = subject_block.fusi_image_pixel_mm
        width_coord, depth_coord = subject_block.fusi_get_coords_inplane_mm()



        ## Code to extract spikes
        spikes_lh = subject_block.ephys_get_probe_spikes_object(probe_name_lh)  # Object
        spikes_rh = subject_block.ephys_get_probe_spikes_object(probe_name_rh)  # Object

        # Extract fUSI and ephys masks
        #############################
        # NOTE:
        # * slice_fusi_mask: Mask of voxels at site of ephys probe insertion
        # * slice_probe_mask: Location of probe depths spanning the fUSI slice
        relative_depth_lh, slice_fusi_mask_lh, slice_probe_mask_lh = subject_block.fusi_get_slice_probedepth_and_voxels(
            probe_name_lh)
        probe_min_depthum_lh, probe_max_depthum_lh = (slice_probe_mask_lh * 1000).astype(int)  # min/max depth of left probe

        relative_depth_rh, slice_fusi_mask_rh, slice_probe_mask_rh = subject_block.fusi_get_slice_probedepth_and_voxels(
            probe_name_rh)
        probe_min_depthum_rh, probe_max_depthum_rh = (slice_probe_mask_rh * 1000).astype(int)  # min/max depth of right probe


        # Build MUA matrix for units intersecting the fUSI slice
        n_mua_units_lh, mua_matrix_lh = spikes_lh.time_locked_mua_matrix(
            fusi_times,
            dt=fusi_dt,
            good_clusters=spikes_lh.good_clusters,
            cluster_depths=spikes_lh.cluster_depths,
            min_depthum=probe_min_depthum_lh,
            max_depthum=probe_max_depthum_lh)

        n_mua_units_rh, mua_matrix_rh = spikes_rh.time_locked_mua_matrix(
            fusi_times,
            dt=fusi_dt,
            good_clusters=spikes_rh.good_clusters,
            cluster_depths=spikes_rh.cluster_depths,
            min_depthum=probe_min_depthum_rh,
            max_depthum=probe_max_depthum_rh)

        ## Code to extract firing rate (V1 and Hippo)

        ## Save the data above at data_location + '\' subjectID +'\' + date + '\' + 'spontaneous' + '\' + block_id
        save_dir = os.path.join(save_location, subjectID, date, 'spontaneous', str(block_id_sp))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'data.mat')

        savemat(save_path, {
            't_PDI': fusi_times,
            'f_dt': fusi_dt,
            'PDI': fusi_data,
            'relative_depth_mask_lh': relative_depth_lh,
            'relative_depth_mask_rh': relative_depth_rh,
            'number_of_sites_lh': n_mua_units_lh,
            'number_of_sites_rh': n_mua_units_rh,
            'spike_rate_lh': mua_matrix_lh,
            'spike_rate_rh': mua_matrix_rh,
            'slice_PDI_mask_lh': slice_fusi_mask_lh,
            'slice_PDI_mask_rh': slice_fusi_mask_rh,
            'probe_min_depth_lh': probe_min_depthum_lh,
            'probe_max_depth_lh': probe_max_depthum_lh,
            'probe_min_depth_rh': probe_min_depthum_rh,
            'probe_max_depth_rh': probe_max_depthum_rh,
            'probe_width': mua_probe_widthum,
            'width_resolution_mm': width_resolution_mm,
            'depth_resolution_mm': depth_resolution_mm,
            'width_coord': width_coord,
            'depth_coord': depth_coord
        })

    ## For checkerboard
    for block_id in checkerboard_blocks:
        subject_block = handler.MetaBlock(subjectID, date, block_id)
        ## Code to extract fUS
        fusi_times, fusi_data = subject_block.fusi_get_data(dt_ms=300, window_ms=400, svddrop=15,
                                                            freq_cutoffhz=15, roi_name=None, mirrored=False,
                                                            verbose=True)
        ## Code to extract spatial resolution and depth width information (in mm)
        width_resolution_mm, depth_resolution_mm = subject_block.fusi_image_pixel_mm
        width_coord, depth_coord = subject_block.fusi_get_coords_inplane_mm()

        ## Code to extract stimulus sequence
        stimulus_start_times, stimulus_end_times = subject_block.stimulus_load_times()
        _, repeat_ids = subject_block.stimulus_checkerboard_times()
        stimulus_sequence = subject_block.timeline.stimulus_sequence

        sorted_start_times = np.zeros(repeat_ids.size)
        sorted_stim_ids = np.zeros(repeat_ids.size)
        sorted_end_times = np.zeros(repeat_ids.size)
        for col_idx in range(repeat_ids.shape[1]):
            if ((col_idx+1) % 2) == 0:
                sorted_start_times[repeat_ids.shape[0]*col_idx:repeat_ids.shape[0]*(1+col_idx)] = np.flip(
                    stimulus_start_times[:, col_idx])
                sorted_stim_ids[repeat_ids.shape[0] * col_idx:repeat_ids.shape[0] * (1 + col_idx)] = np.flip(
                    repeat_ids[:, col_idx])
                sorted_end_times[repeat_ids.shape[0] * col_idx:repeat_ids.shape[0] * (1 + col_idx)] = np.flip(
                    stimulus_end_times[:, col_idx])
            else:
                sorted_start_times[repeat_ids.shape[0] * col_idx:repeat_ids.shape[0]*(1+col_idx)]\
                    = stimulus_start_times[:, col_idx]
                sorted_stim_ids[repeat_ids.shape[0] * col_idx:repeat_ids.shape[0] * (1 + col_idx)]\
                    = repeat_ids[:, col_idx]
                sorted_end_times[repeat_ids.shape[0] * col_idx:repeat_ids.shape[0] * (1 + col_idx)] \
                    = stimulus_end_times[:, col_idx]

        stim_start = []
        stim_start.append(sorted_start_times[sorted_stim_ids == 0])  # Left
        stim_start.append(sorted_start_times[sorted_stim_ids == 1])  # Center
        stim_start.append(sorted_start_times[sorted_stim_ids == 3])  # Right
        stim_start = np.array(stim_start).T

        stim_end = []
        stim_end.append(sorted_end_times[sorted_stim_ids == 0])  # Left
        stim_end.append(sorted_end_times[sorted_stim_ids == 1])  # Center
        stim_end.append(sorted_end_times[sorted_stim_ids == 3])  # Right
        stim_end = np.array(stim_end).T

        ## Align stimulus to fUS measurement
        stim = np.zeros([fusi_times.size, 3])

        # Loop through each onset/offset pair and set stimulus to 1 in that interval
        for col in range(stim_start.shape[1]):
            for onset, offset in zip(stim_start[:, col], stim_end[:, col]):
                stim[(fusi_times >= onset) & (fusi_times < offset), col] = 1



        ## Code to extract spikes
        spikes_lh = subject_block.ephys_get_probe_spikes_object(probe_name_lh)  # Object
        spikes_rh = subject_block.ephys_get_probe_spikes_object(probe_name_rh)  # Object

        # Extract fUSI and ephys masks
        #############################
        # NOTE:
        # * slice_fusi_mask: Mask of voxels at site of ephys probe insertion
        # * slice_probe_mask: Location of probe depths spanning the fUSI slice
        relative_depth_lh, slice_fusi_mask_lh, slice_probe_mask_lh = subject_block.fusi_get_slice_probedepth_and_voxels(
            probe_name_lh)
        probe_min_depthum_lh, probe_max_depthum_lh = (slice_probe_mask_lh * 1000).astype(
            int)  # min/max depth of left probe

        relative_depth_rh, slice_fusi_mask_rh, slice_probe_mask_rh = subject_block.fusi_get_slice_probedepth_and_voxels(
            probe_name_rh)
        probe_min_depthum_rh, probe_max_depthum_rh = (slice_probe_mask_rh * 1000).astype(
            int)  # min/max depth of right probe

        # Build MUA matrix for units intersecting the fUSI slice
        n_mua_units_lh, mua_matrix_lh = spikes_lh.time_locked_mua_matrix(
            fusi_times,
            dt=fusi_dt,
            good_clusters=spikes_lh.good_clusters,
            cluster_depths=spikes_lh.cluster_depths,
            min_depthum=probe_min_depthum_lh,
            max_depthum=probe_max_depthum_lh)

        n_mua_units_rh, mua_matrix_rh = spikes_rh.time_locked_mua_matrix(
            fusi_times,
            dt=fusi_dt,
            good_clusters=spikes_rh.good_clusters,
            cluster_depths=spikes_rh.cluster_depths,
            min_depthum=probe_min_depthum_rh,
            max_depthum=probe_max_depthum_rh)

        ## Code to extract firing rate (V1 and Hippo)

        ## Save the data above at data_location + '\' subjectID +'\' + date + '\' + 'checkerboard' + '\' + block_id
        save_dir = os.path.join(save_location, subjectID, date, 'checkerboard', str(block_id))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'data.mat')

        savemat(save_path, {
            'stim': stim,
            't_PDI': fusi_times,
            'f_dt': fusi_dt,
            'PDI': fusi_data,
            'relative_depth_mask_lh': relative_depth_lh,
            'relative_depth_mask_rh': relative_depth_rh,
            'number_of_sites_lh': n_mua_units_lh,
            'number_of_sites_rh': n_mua_units_rh,
            'spike_rate_lh': mua_matrix_lh,
            'spike_rate_rh': mua_matrix_rh,
            'slice_PDI_mask_lh': slice_fusi_mask_lh,
            'slice_PDI_mask_rh': slice_fusi_mask_rh,
            'probe_min_depth_lh': probe_min_depthum_lh,
            'probe_max_depth_lh': probe_max_depthum_lh,
            'probe_min_depth_rh': probe_min_depthum_rh,
            'probe_max_depth_rh': probe_max_depthum_rh,
            'probe_width': mua_probe_widthum,
            'width_resolution_mm': width_resolution_mm,
            'depth_resolution_mm': depth_resolution_mm,
            'width_coord': width_coord,
            'depth_coord': depth_coord
        })


