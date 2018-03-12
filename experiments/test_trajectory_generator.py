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

# from gym_urbandriving.planner.trajectory import Trajectory

import glob
import cPickle as pickle

cnfg = Config()
vr = VizRegistration(cnfg)
hm = Homography(cnfg)
of = ObsFiltering(cnfg)
ta = TrajectoryAnalysis(cnfg)

def sample_trajectory(start_lane_index, primitive, traj_generator_model, sigma=150, mixing_param=0.1):
    mean, cov = traj_generator_model[start_lane_index][primitive]
    sample = np.random.multivariate_normal(mean, mixing_param * cov + (1 - mixing_param) * np.eye(8))

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

def visualize_trajectory(x_sampled, y_sampled, x_traj, y_traj, x_uds, y_uds):
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
    plt.plot(x_uds, y_uds, 'r')
    plt.show()

def compute_distance(x1, y1, x2, y2, num=50):
    x1_sampled = []
    y1_sampled = []
    x2_sampled = []
    y2_sampled = []

    increment_seq1 = float(len(x1) - 1) / (num - 1)
    increment_seq2 = float(len(x2) - 1) / (num - 1)

    for i in range(num):
        x1_sampled.append(x1[int(increment_seq1 * i)])
        y1_sampled.append(y1[int(increment_seq1 * i)])
        x2_sampled.append(x2[int(increment_seq2 * i)])
        y2_sampled.append(y2[int(increment_seq2 * i)])

    x1_sampled = np.array(x1_sampled)
    y1_sampled = np.array(y1_sampled)
    x2_sampled = np.array(x2_sampled)
    y2_sampled = np.array(y2_sampled)

    return np.mean(np.sqrt(np.square(x1_sampled - x2_sampled) + np.square(y1_sampled - y2_sampled)))

def evaluate(held_out_traj, uds_points, traj_generator_model):
    start_lane_index, _ = traj.get_start_lane_index()
    primitive = ta.get_trajectory_primitive(traj)

    uds_x, uds_y = process(uds_points[0], uds_points[1])

    x_traj, y_traj = traj.get_smoothed_polynomial_points(sigma=150)

    x_sampled, y_sampled = sample_trajectory(start_lane_index, primitive, traj_generator_model)

    # visualize_trajectory(x_traj, y_traj, x_sampled, y_sampled)

    x_traj_cropped, y_traj_cropped = process(x_traj, y_traj)

    x_sampled_cropped, y_sampled_cropped = process(x_sampled, y_sampled)

    # visualize_trajectory(x_sampled_cropped, y_sampled_cropped, x_traj_cropped, y_traj_cropped, uds_x, uds_y)

    return compute_distance(x_traj_cropped, y_traj_cropped, uds_x, uds_y), compute_distance(x_sampled_cropped, y_sampled_cropped, x_traj_cropped, y_traj_cropped)

def process(xs, ys):
    first_index = None

    for i in range(len(xs)):
        if valid_point(xs[i], ys[i]):
            first_index = i
            # print "First Index", first_index
            break


    last_index = None
    for i in range(len(xs) - 1, -1, -1):
        if valid_point(xs[i], ys[i]):
            last_index = i
            # print "Last Index", last_index
            break

    if first_index is None:
        first_index = 0

    if last_index is None:
        last_index = len(xs) - 1

    return xs[first_index:last_index + 1], ys[first_index:last_index + 1]

def valid_point(x, y):
    x_min = 300
    x_max = 700
    y_min = 300
    y_max = 700

    return x >= x_min and x <= x_max and y >= y_min and y <= y_max

traj_generator_model = pickle.load(open('{0}/{1}'.format(cnfg.save_debug_pickles_path, 'traj_generator_model.pkl'), 'r'))
held_out_traj_dict = pickle.load(open('{0}/{1}'.format(cnfg.save_debug_pickles_path, 'held_out_traj_by_primitive.pkl'), 'r'))
uds_traj_dict = pickle.load(open('{0}/{1}'.format(cnfg.save_debug_pickles_path, 'traj_uds_samples.pkl'), 'r'))

all_held_out_distances = {
        1: {'left': [], 'forward': [], 'right': []},
        3: {'left': [], 'forward': [], 'right': []},
        5: {'left': [], 'forward': [], 'right': []},
        7: {'left': [], 'forward': [], 'right': []}
    }

all_uds_distances = {
        1: {'left': [], 'forward': [], 'right': []},
        3: {'left': [], 'forward': [], 'right': []},
        5: {'left': [], 'forward': [], 'right': []},
        7: {'left': [], 'forward': [], 'right': []}
    }

bad_cases = [(1, 'forward', 0), (1, 'forward', 4), (5, 'forward', 0)]

# uds_distances = []
# held_out_distances = []

for i in [1, 3, 5, 7]:
    for j in ['left', 'forward', 'right']:
        uds_distances = []
        held_out_distances = []

        for k in range(5):
            if (i, j, k) not in bad_cases:

                traj = held_out_traj_dict[i][j][k]

                points_uds = np.array(uds_traj_dict[i][j][k]).T

                uds_dist, held_out_dist = evaluate(traj, points_uds, traj_generator_model)

                # if uds_dist < 50 and held_out_dist < 50:
                #     uds_distances.append(uds_dist)
                #     held_out_distances.append(held_out_dist)

                uds_distances.append(uds_dist)
                held_out_distances.append(held_out_dist)

        print i, j
        print "UDS:", np.mean(uds_distances), np.std(uds_distances) / np.sqrt(len(uds_distances))#, uds_distances
        print "Generator:", np.mean(held_out_distances), np.std(held_out_distances) / np.sqrt(len(held_out_distances))#, held_out_distances

        # all_held_out_distances[i][j] = np.mean(held_out_distances)
        # all_uds_distances[i][j] = np.mean(uds_distances)

            # uds_x, uds_y = process(points_uds[0], points_uds[1])

            # x_traj, y_traj = traj.get_smoothed_polynomial_points()
            
            # cropped_x_traj, cropped_y_traj = process(x_traj, y_traj)

            # visualize_trajectory(uds_x, uds_y, cropped_x_traj, cropped_y_traj)

# print(np.mean(uds_distances))
# print(np.mean(held_out_distances))

# print "Distances on Held Out Set by Primitive:", all_held_out_distances
# print "Distances on UDS Set by Primitive:", all_uds_distances


# for i in range(5):
#     compute_distance(held_out_traj_dict[3]['left'][0], traj_generator_model)

# start_lane_indexes = [1, 3, 5, 7]
# primitives = ['left', 'forward', 'right']

# for start_lane_index in start_lane_indexes:
#     for primitive in primitives:
#         print start_lane_index, primitive

#         for i in range(3):
#             print(compute_distance(held_out_traj_dict[start_lane_index][primitive][i], traj_generator_model))

# Hardcoded Goal States: (550, 100) [Lane 2], (450, 900) [Lane 6], (900, 550) [Lane 0], (100, 450) [Lane 4]
# Hardcoded Start States: (900, 450) [Lane 1], (450, 100) [Lane 3], (100, 550) [Lane 5], (550, 100) [Lane 8]

# traj_dict = {
#         1: {'left': [], 'forward': [], 'right': []},
#         3: {'left': [], 'forward': [], 'right': []},
#         5: {'left': [], 'forward': [], 'right': []},
#         7: {'left': [], 'forward': [], 'right': []}
#     }

# # data = np.load("./../Urban_Driving_Simulator/test_data/TCP/rollout_0.npy")

# # print(data)

# # print(data[1]['goal_states_selected'])
# # print(data[1]['start_states_selected'])

# lane_index_to_position = { 2 : (550, 100), 6 : (450, 900), 0 : (900, 550), 4 : (100, 450),
#                            1 : (900, 450), 3 : (450, 100), 5 : (100, 550), 7 : (550, 100) }

# goal_states = { 2 : (550, 100), 6 : (450, 900), 0 : (900, 550), 4 : (100, 450) }

# start_lane_index_conversion = { 0 : 3, 3 : 5,  1 : 7, 2 : 1 }

# primitive_conversion = { (1, 2) : 'right', (1, 4) : 'forward', (1, 6) : 'left', 
#                          (3, 4) : 'right', (3, 6) : 'forward', (3, 0) : 'left',
#                          (5, 6) : 'right', (5, 0) : 'forward', (5, 2) : 'left',
#                          (7, 0) : 'right', (7, 2) : 'forward', (7, 4) : 'left'  }

# for j in range(40):
#     data = np.load("./../Urban_Driving_Simulator/test_data/TCP/rollout_{0}.npy".format(j))
#     for i in range(len(data[1]['start_states_selected'])):
#         uds_start_state = data[1]['start_states_selected'][i]
#         tcp_start_state = start_lane_index_conversion[uds_start_state]

#         traj = data[1]['geo_trajs'][i]

#         points = traj.get_renderable_points()
#         last_point = points[-1]

#         min_dist = 100000.0
#         goal = -1
#         for i in goal_states:
#             target_x, target_y = goal_states[i]

#             dist = (target_x - last_point[0]) ** 2 + (target_y - last_point[1]) ** 2

#             if dist < min_dist:
#                 min_dist = dist
#                 goal = i

#         # print points[0], points[-1]
#         # print tcp_start_state, goal

#         primitive = None

#         if (tcp_start_state, goal) in primitive_conversion:
#             # print primitive_conversion[(tcp_start_state, goal)]
#             primitive = primitive_conversion[(tcp_start_state, goal)]

#             traj_dict[tcp_start_state][primitive].append(points)

# for i in [1, 3, 5, 7]:
#     for j in ['left', 'forward', 'right']:
#         print len(traj_dict[i][j])

# with open(os.path.join(cnfg.generator_pickles_path, 'traj_uds_samples_all.pkl'), 'w+') as pkl_file:
#     pickle.dump(traj_dict, pkl_file)

    # points = traj.get_renderable_points()
    # print(points[0], points[-1])
