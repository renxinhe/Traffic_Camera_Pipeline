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

def sample_trajectory(start_lane_index, primitive, traj_generator_model, sigma=75):
    mean, cov = traj_generator_model[start_lane_index][primitive]
    sample = np.random.multivariate_normal(mean, cov)

    x_coeffs = sample[:4]
    y_coeffs = sample[4:]

    u_new = np.linspace(0.0, 1.0, 1000)

    x_poly = np.poly1d(x_coeffs)
    y_poly = np.poly1d(y_coeffs)

    x_new = x_poly(u_new)
    y_new = y_poly(u_new)
    x_new = gaussian_filter1d(x_new, sigma)
    y_new = gaussian_filter1d(y_new, sigma)

    return x_new, y_new

def visualize_trajectory(x_traj, y_traj, x_sampled, y_sampled):
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

    plt.plot(x_traj, y_traj, 'b')
    plt.plot(x_sampled, y_sampled, 'g')
    plt.show()

def compute_distance(traj, traj_generator_model):
    start_lane_index, _ = traj.get_start_lane_index()
    primitive = ta.get_trajectory_primitive(traj)

    x_traj, y_traj = traj.get_smoothed_polynomial_points()

    x_sampled, y_sampled = sample_trajectory(start_lane_index, primitive, traj_generator_model)

    visualize_trajectory(x_traj, y_traj, x_sampled, y_sampled)

    return np.linalg.norm(np.concatenate([x_traj, y_traj]) - np.concatenate([x_sampled, y_sampled]))

traj_generator_model = pickle.load(open('{0}/{1}'.format(cnfg.save_debug_pickles_path, 'traj_generator_model.pkl'), 'r'))
held_out_traj_dict = pickle.load(open('{0}/{1}'.format(cnfg.save_debug_pickles_path, 'held_out_traj_by_primitive.pkl'), 'r'))

# for _ in range(5):
#     sample_trajectory(5, 'forward', traj_generator_model)

for i in range(5):
    print(compute_distance(held_out_traj_dict[7]['left'][i], traj_generator_model))
