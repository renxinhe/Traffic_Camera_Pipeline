#/home/jren/env/bin/python
from __future__ import unicode_literals

import sys, os, pdb
import time
import cv2
import youtube_dl
import numpy as np
import tensorflow as tf
import cPickle as pickle
from urlparse import parse_qs

from AbstractDetector import AbstractDetector
from ssd_detector import SSD_VGG16Detector

import tcp.object_detection.Re3 as Re3
sys.path.insert(0, os.path.dirname(Re3.__file__))
from Re3.tracker import re3_tracker

from tcp.object_detection.cropper import Cropper

from tcp.object_detection.init_labeler import InitLabeler
from tcp.object_detection.init_labeler_opencv import InitLabeler_OpenCV
from tcp.object_detection.visualization import bboxes_draw_on_img, colors_original, colors_dark

from tcp.utils.utils import bbox_IoU, normalize_bbox, denormalize_bbox, bbox_near_margin


class VideoLabeler():

    def __init__(self, config):

        self.config = config
        self.cropper = Cropper(self.config)
        self.ssd_detector = SSD_VGG16Detector('ssd_vgg16', self.config.ssd_checkpoint_path, cropper=self.cropper)

    def __del__(self):
        self.close_video()

    def load_video(self, video_path):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.ssd_detector.setStreamURL(video_path)
        if not AbstractDetector.openCapture(self.ssd_detector):
            raise ValueError('Video file %s failed to open.' % video_path)
        print 'Scanning %s...' % (video_path)

    def close_video(self):
        if self.ssd_detector.cap.isOpened():
            self.ssd_detector.cap.release()

    def generate_bounding_boxes(self, debug_pickle=False):
        ### First pass: get bounding ###
        all_rclasses = []
        all_rbboxes = []

        if debug_pickle:
            try:
                all_rclasses = pickle.load(open('{0}/{1}/{1}_classes.cpkl'.format(self.config.save_debug_pickles_path, self.video_name), 'r'))
                print 'Loaded "{0}/{1}/{1}_classes.cpkl".'.format(self.config.save_debug_pickles_path, self.video_name)
            except IOError as e:
                print 'Unable to load "{0}/{1}/{1}_classes.cpkl"'.format(self.config.save_debug_pickles_path, self.video_name)
            try:
                all_rbboxes = pickle.load(open('{0}/{1}/{1}_bboxes.cpkl'.format(self.config.save_debug_pickles_path, self.video_name), 'r'))
                print 'Loaded "{0}/{1}/{1}_bboxes.cpkl".'.format(self.config.save_debug_pickles_path, self.video_name)
            except IOError as e:
                print 'Unable to load "{0}/{1}/{1}_bboxes.cpkl"'.format(self.config.save_debug_pickles_path, self.video_name)

        if all_rclasses == [] or all_rbboxes == []:
            print 'Some detection pickle file failed to load. Running detector network... This may take a while.'
            while self.ssd_detector.cap.isOpened():
                ret, frame = self.ssd_detector.cap.read()
                if frame is None:
                    break
                rclasses, rscores, rbboxes = self.ssd_detector.get_bounding_box(frame)

                # Filter bboxes with cropper mask
                rclasses_copy = list(rclasses)
                rclasses = [rclasses[i] for i, bbox in enumerate(rbboxes) if self.cropper.check_is_valid(rclasses_copy[i], *bbox)]
                rbboxes = [bbox for i, bbox in enumerate(rbboxes) if self.cropper.check_is_valid(rclasses_copy[i], *bbox)]

                all_rclasses.append(rclasses)
                all_rbboxes.append(rbboxes)
            self.ssd_detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if debug_pickle:
            if not os.path.exists('{0}/{1}'.format(self.config.save_debug_pickles_path, self.video_name)):
                os.makedirs('{0}/{1}'.format(self.config.save_debug_pickles_path, self.video_name))
            pickle.dump(all_rclasses, open('{0}/{1}/{1}_classes.cpkl'.format(self.config.save_debug_pickles_path, self.video_name), 'w+'))
            pickle.dump(all_rbboxes, open('{0}/{1}/{1}_bboxes.cpkl'.format(self.config.save_debug_pickles_path, self.video_name), 'w+'))

        self.all_rbboxes = all_rbboxes
        self.all_rclasses = all_rclasses
        return self.all_rclasses, self.all_rbboxes

    def run_init_labeler(self, all_rbboxes=None, all_rclasses=None, show_gui=True, debug_pickle=False):
        all_rbboxes = self.all_rbboxes if all_rbboxes is None else all_rbboxes
        all_rclasses = self.all_rclasses if all_rclasses is None else all_rclasses

        start_time = time.time()
        self.init_labeler = InitLabeler_OpenCV(self.config, self.ssd_detector.cap, all_rbboxes, all_rclasses,
                                        video_name=self.video_name, cache_frames=True, show_gui=show_gui)
        elapsed_time = time.time() - start_time

        if debug_pickle:
            with open('{0}/{1}/{1}_timing.txt'.format(self.config.save_debug_pickles_path, self.video_name),'a+') as timing_file:
                timing_file.write('InitLabeler timing: %d min %d sec (%d seconds)\n\n' % (elapsed_time // 60, elapsed_time % 60, elapsed_time))
            print 'InitLabeler timing: %d min %d sec\n' % (elapsed_time // 60, elapsed_time % 60)

        self.all_rbboxes = self.init_labeler.all_rbboxes
        self.all_rclasses = self.init_labeler.all_rclasses
        return self.all_rclasses, self.all_rbboxes

    def generate_trajectories(self,
                              all_rbboxes=None,
                              all_rclasses=None,
                              output_limit=None,
                              num_skip_frames=1,
                              save_images=False):
        frame_i = 0
        frame_skip = 0

        all_rbboxes = self.all_rbboxes if all_rbboxes is None else all_rbboxes
        all_rclasses = self.all_rclasses if all_rclasses is None else all_rclasses

        trajectory = []

        ### Second pass: process bounding box data ###
        self.ssd_detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.ssd_detector.cap.isOpened():
            ret, frame = self.ssd_detector.cap.read()
            if frame is None:
                break

            frame_skip += 1

            assert num_skip_frames > 0 and int(num_skip_frames) == num_skip_frames
            frame_skip %= num_skip_frames
            if frame_skip != 0:
                continue

            # Process frame here
            rclasses = all_rclasses[frame_i]
            rbboxes = all_rbboxes[frame_i]

            if save_images:
                debug_img_path = os.path.join(self.config.save_debug_img_path, self.video_name)
                if not os.path.exists(debug_img_path):
                    os.makedirs(debug_img_path)
                bboxes_draw_on_img(frame, rclasses, [1.0] * len(rclasses), rbboxes, colors_dark)
                cv2.imwrite(os.path.join(debug_img_path, '%s_%07d.jpg' % (self.video_name, frame_i)), frame)

            unique, counts = np.unique(rclasses, return_counts=True)
            classes_counts = dict(zip(unique, counts))
            car_count = classes_counts.get(7)

            if frame_i % 100 == 0:
                if output_limit is None:
                    total_frames = len(all_rclasses)
                else:
                    total_frames = min(output_limit, len(all_rclasses))
                print "Processed frames: %d/%d " % (frame_i, total_frames)

            current_frame = []

            car_cords = self.get_car_cords(frame_i, rclasses, rbboxes)
            ped_cords = self.get_pedestrian_cords(frame_i, rclasses, rbboxes)

            frame_i += 1
            if output_limit is not None and frame_i > output_limit:
                break

            for car_cord in car_cords:
                current_frame.append(car_cord)

            if self.config.use_pedestrian:
                for ped_cord in ped_cords:
                    current_frame.append(ped_cord)

            if len(current_frame) != 0:
                trajectory.append(current_frame)

        print 'Done processing %d frames.' % (frame_i - 1)
        self.ssd_detector.cap.release()

        return trajectory

    def generate_re3_trajectories(self,
                                  all_rbboxes=None,
                                  all_rclasses=None,
                                  threshold_IoU=0.7,
                                  threshold_min_frames=30,
                                  debug_pickle=False,
                                  save_images=False):
        frame_i = 0

        all_rbboxes = self.all_rbboxes if all_rbboxes is None else all_rbboxes
        all_rclasses = self.all_rclasses if all_rclasses is None else all_rclasses
        assert len(all_rbboxes) == len(all_rclasses)

        if debug_pickle:
            try:
                trajectories_dict = pickle.load(open('{0}/{1}/{1}_trajectories.cpkl'.format(self.config.save_debug_pickles_path, self.video_name), 'r'))
                print 'Loaded "{0}/{1}/{1}_trajectories.cpkl".'.format(self.config.save_debug_pickles_path, self.video_name)
                return trajectories_dict
            except IOError as e:
                print 'Unable to load "{0}/{1}/{1}_trajectories.cpkl"'.format(self.config.save_debug_pickles_path, self.video_name)
                print 'Running tracking network... This may take a while.'

        trajectories_dict = {}

        tf.reset_default_graph()
        tracker = re3_tracker.Re3Tracker(self.config)
        traj_ids = []
        traj_counts = {}

        self.ssd_detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.ssd_detector.cap.isOpened():
            ret, frame = self.ssd_detector.cap.read()
            if frame is None:
                break

            # Process frame here
            rbboxes = all_rbboxes[frame_i]
            rclasses = all_rclasses[frame_i]

            if save_images:
                debug_img_path = os.path.join(self.config.save_debug_img_path, self.video_name, 'SSD')
                if not os.path.exists(debug_img_path):
                    os.makedirs(debug_img_path)
                assert len(rbboxes) == len(rclasses)
                bboxes_draw_on_img(frame, rclasses, [''] * len(rbboxes), rbboxes, colors_dark)
                cv2.imwrite(os.path.join(debug_img_path, '%s_%07d.jpg' % (self.video_name, frame_i)), frame)

            rbboxes_in_frame_i = []
            rclasses_in_frame_i = []

            traj_ids_keep = []
            if frame_i % 100 == 0:
                print '\nProcessing frame %d' % frame_i
            for traj_id in traj_ids:
                tracked_bbox = tracker.track(traj_id, frame)

                # Adding new bbox to output dictionary
                trajectory_dict = trajectories_dict[traj_id]
                trajectory_bboxes = trajectory_dict['bboxes']
                trajectory_bboxes.append(tracked_bbox)

                denormed_tracked_bbox = list(tracked_bbox)
                # print 'tracker tracked_bbox', traj_id, tracked_bbox
                tracked_bbox = normalize_bbox(tracked_bbox, self.config.img_dim)

                # Find index of tracked bboxes to remove
                rbboxes_new_track_index = list(range(len(rbboxes)))
                for i, rbbox in enumerate(rbboxes):
                    if bbox_IoU(rbbox, tracked_bbox) >= threshold_IoU:
                        if i in rbboxes_new_track_index:
                            rbboxes_new_track_index.remove(i)
                rbboxes = np.array(rbboxes)[rbboxes_new_track_index].tolist()
                rclasses = np.array(rclasses)[rbboxes_new_track_index].tolist()

                # Remove tracks outside of crop zone
                traj_rclass = traj_id.split('_')[0]
                if not bbox_near_margin(denormed_tracked_bbox, self.config.img_dim, margin=10)\
                   and self.cropper.check_is_valid(traj_rclass, *tracked_bbox):
                    traj_ids_keep.append(traj_id)
                    rbboxes_in_frame_i.append(tracked_bbox)
                    rclasses_in_frame_i.append(traj_rclass)
            traj_ids = traj_ids_keep
            # print 'rbboxes', rbboxes

            # Track new bboxes
            for i, rbbox in enumerate(rbboxes):
                rclass = rclasses[i]
                if rclass in traj_counts:
                    traj_counts[rclass] += 1
                else:
                    traj_counts[rclass] = 1
                new_traj_id = '%s_%s' % (rclass, traj_counts[rclass])
                traj_ids.append(new_traj_id)

                denormed_rbbox = denormalize_bbox(rbbox, self.config.img_dim)
                # print 'new tracked_bbox', new_traj_id, denormed_rbbox
                trajectories_dict[new_traj_id] = {'start_frame': frame_i,
                                                  'bboxes': [denormed_rbbox]}

                tracker.track(new_traj_id, frame, denormed_rbbox)
                rbboxes_in_frame_i.append(rbbox)
                rclasses_in_frame_i.append(rclass)

            if save_images:
                debug_img_path = os.path.join(self.config.save_debug_img_path, self.video_name)
                if not os.path.exists(debug_img_path):
                    os.makedirs(debug_img_path)
                assert len(rbboxes_in_frame_i) == len(rclasses_in_frame_i)
                bboxes_draw_on_img(frame, rclasses_in_frame_i, traj_ids, rbboxes_in_frame_i, colors_original)
                cv2.imwrite(os.path.join(debug_img_path, '%s_%07d.jpg' % (self.video_name, frame_i)), frame)

            frame_i += 1

        for traj_id, trajectory_dict in trajectories_dict.items():
            if len(trajectory_dict['bboxes']) < threshold_min_frames:
                del(trajectories_dict[traj_id])
        return trajectories_dict


    def get_car_cords(self, frame_i, rclasses, rbboxes):
        rclasses = np.array(rclasses)
        rbboxes = np.array(rbboxes)
        assert len(rclasses) == len(rbboxes), 'rclasses: %s\n rbboxes: %s' % (rclasses, rbboxes)

        frame = []
        assert self.init_labeler is not None, 'Call run_init_labeler before generate_trajectories'
        arg_init_label = self.init_labeler.get_arg_init_label(frame_i)
        for i in range(len(rclasses)):
            # 6: bus, 7: car, 14: motorcycle
            if int(rclasses[i]) in [6, 7, 14]:
                x_min, y_min, x_max, y_max = rbboxes[i]
                point = {'x': (x_min + x_max) / 2.0,
                         'y': y_max,
                         'cls_label': 'car',
                         't': frame_i}
                if i in arg_init_label:
                    point['is_initial_state'] = True
                else:
                    point['is_initial_state'] = False
                frame.append(point)
        return frame


    def get_pedestrian_cords(self, frame_i, rclasses, rbboxes):
        rclasses = np.array(rclasses)
        rbboxes = np.array(rbboxes)
        assert len(rclasses) == len(rbboxes), 'rclasses: %s\n rbboxes: %s' % (rclasses, rbboxes)

        frame = []
        assert self.init_labeler is not None, 'Call run_init_labeler before generate_trajectories'
        arg_init_label = self.init_labeler.get_arg_init_label(frame_i)
        for i in range(len(rclasses)):
            # 2: bicyle, 15: person
            if int(rclasses[i]) in [15]:
                x_min, y_min, x_max, y_max = rbboxes[i]
                point = {'x': (x_min + x_max) / 2.0,
                         'y': (y_min + y_max) / 2.0,
                         'cls_label': 'pedestrian',
                         't': frame_i}
                if i in arg_init_label:
                    point['is_initial_state'] = True
                else:
                    point['is_initial_state'] = False
                frame.append(point)

        return frame
