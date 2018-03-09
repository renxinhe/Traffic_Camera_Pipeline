from __future__ import unicode_literals

import sys, os
import cv2

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from tcp.registration.homography import Homography
from tcp.registration.obs_filtering import ObsFiltering
from tcp.registration.viz_registration import VizRegistration
from tcp.registration.trajectory_analysis import TrajectoryAnalysis
from tcp.configs.alberta_config import Config

import glob
import cPickle as pickle

cnfg = Config()
vr = VizRegistration(cnfg)
hm = Homography(cnfg)
of = ObsFiltering(cnfg)
ta = TrajectoryAnalysis(cnfg)

###GET VIDEOS
VIDEO_FILE = '%s/*.mp4' % cnfg.video_root_dir
videos = glob.glob(VIDEO_FILE)

### Analyze Videos
traj_dict = {
        1: {'left': [], 'forward': [], 'right': [], 'stopped': []},
        3: {'left': [], 'forward': [], 'right': [], 'stopped': []},
        5: {'left': [], 'forward': [], 'right': [], 'stopped': []},
        7: {'left': [], 'forward': [], 'right': [], 'stopped': []}
    }

held_out_traj_dict = {
        1: {'left': [], 'forward': [], 'right': [], 'stopped': []},
        3: {'left': [], 'forward': [], 'right': [], 'stopped': []},
        5: {'left': [], 'forward': [], 'right': [], 'stopped': []},
        7: {'left': [], 'forward': [], 'right': [], 'stopped': []}
    }

# Visualize trajectory in UDS based on polynomial fit coefficients.
def visualize_trajectory(x_coeffs, y_coeffs, sigma=75):
    u_new = np.linspace(0.0, 1.0, 1000)

    x_poly = np.poly1d(x_coeffs)
    y_poly = np.poly1d(y_coeffs)

    x_new = x_poly(u_new)
    y_new = y_poly(u_new)
    x_new = gaussian_filter1d(x_new, sigma)
    y_new = gaussian_filter1d(y_new, sigma)
    
    if x_new is not None and y_new is not None:
        plt.figure(figsize=(8, 8))
        axes = plt.gca()
        axes.set_xlim([-100,1100])
        axes.set_ylim([-100,1100])
        plt.gca().invert_yaxis()

        plt.plot([400, 400], [0, 1000], color='k')
        plt.plot([500, 500], [0, 1000], color='k')
        plt.plot([600, 600], [0, 1000], color='k')

        plt.plot([0, 1000], [400, 400], color='k')
        plt.plot([0, 1000], [500, 500], color='k')
        plt.plot([0, 1000], [600, 600], color='k')

        # plt.scatter(trajectory.xs, trajectory.ys, c='r', marker='.')
        plt.plot(x_new, y_new)
        plt.show()

for video_path in sorted(videos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    datestamp = video_name.split('_')[-2]
    timestamp = video_name.split('_')[-1]

    year, month, date = [int(i) for i in datestamp.split('-')]
    hour, minute, second = [int(i) for i in timestamp.split('-')]

    # Setting first video
    tmp_time = int('%02d%02d%02d' % (date, hour, minute))
    if tmp_time < 270900:
        continue
    # Setting last video
    if tmp_time > 271300:
    # if tmp_time > 270905:
        break

    print 'Analyzing video: %s' % video_path

    camera_view_trajectory_pickle = '{0}/{1}/{1}_trajectories.cpkl'.format(cnfg.save_debug_pickles_path, video_name)
    camera_view_trajectory = pickle.load(open(camera_view_trajectory_pickle,'r'))

    assert camera_view_trajectory is not None, "%s doesn't have a trajectories pickle file" % video_name

    simulator_view_trajectory = hm.transform_trajectory(camera_view_trajectory)
    filtered_trajectory = of.heuristic_label(simulator_view_trajectory)
    
    for i, traj in enumerate(filtered_trajectory):
        # if i == 0 or i == 1:
        # print("Trajectory", i)

        start_lane_index, _ = traj.get_start_lane_index()
        primitive = ta.get_trajectory_primitive(traj)

        # x_coeffs, y_coeffs = traj.fit_to_polynomial()

        if start_lane_index is not None and\
            traj_dict.get(start_lane_index) is not None:

            if primitive is not None and\
                traj_dict[start_lane_index].get(primitive) is not None:

                traj_dict[start_lane_index][primitive].append(traj)

                # x_coeffs, y_coeffs = traj.fit_to_polynomial()
                # visualize_trajectory(x_coeffs, y_coeffs)

# Save all trajectories grouped by primitive action.
with open(os.path.join(cnfg.save_debug_pickles_path, 'traj_by_primitive.pkl'), 'w+') as pkl_file:
    pickle.dump(traj_dict, pkl_file)

# Number of trajectories to hold out for each primitive.
NUM_HELD_OUT = 5

traj_generator_model = {}

for start_lane_index in traj_dict:

    lane_index_model = {}

    for primitive in traj_dict[1]:
        trajectories = traj_dict[start_lane_index][primitive]
        num_trajectories = len(trajectories)

        held_out = np.random.choice(num_trajectories, NUM_HELD_OUT, replace=False)
        print(held_out)

        all_params = []

        for i in range(num_trajectories):
            traj = trajectories[i]

            if i in held_out:
                held_out_traj_dict[start_lane_index][primitive].append(traj)
            else:
                x_coeffs, y_coeffs = traj.fit_to_polynomial()
                combined = np.concatenate([x_coeffs, y_coeffs])
                all_params.append(combined)

        print(np.array(all_params).shape)

        mean = np.mean(np.array(all_params), axis=0)
        cov = np.cov(np.array(all_params).T)

        lane_index_model[primitive] = (mean, cov)

    traj_generator_model[start_lane_index] = lane_index_model

print(traj_generator_model)

# Save all held out trajectories grouped by primitive action.
with open(os.path.join(cnfg.save_debug_pickles_path, 'held_out_traj_by_primitive.pkl'), 'w+') as pkl_file:
    pickle.dump(held_out_traj_dict, pkl_file)

# Save trajectory generator model grouped by primitive action.
with open(os.path.join(cnfg.save_debug_pickles_path, 'traj_generator_model.pkl'), 'w+') as pkl_file:
    pickle.dump(traj_generator_model, pkl_file)
