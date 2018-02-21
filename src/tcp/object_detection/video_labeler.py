#/home/jren/env/bin/python
from __future__ import unicode_literals

import sys, os
import cv2
import youtube_dl
import numpy as np
from urlparse import parse_qs

from AbstractDetector import AbstractDetector
from ssd_detector import SSD_VGG16Detector 
from tcp.object_detection.cropper import Cropper

import IPython

from tcp.object_detection.init_labeler import InitLabeler


class LabelVideo():

    def __init__(self, config, net_path=None):

        self.config = config
        self.cropper = Cropper(self.config)

        self.ssd_detector = SSD_VGG16Detector('ssd_vgg16', self.config.check_point_path, cropper=self.cropper)
        self.t = 0

    def label_video(self, video_path, output_limit=500, num_skip_frames=1):
            self.ssd_detector.setStreamURL(video_path)
            if not AbstractDetector.openCapture(self.ssd_detector):
                raise ValueError('Video file %s failed to open.' % video_path)
            print 'Scanning %s...' % (video_path)


            ### First pass: get bounding ###
            all_rclasses = []
            all_rscores = []
            all_rbboxes = []
            while self.ssd_detector.cap.isOpened():
                ret, frame = self.ssd_detector.cap.read()
                if frame is None:
                    break
                rclasses, rscores, rbboxes = self.ssd_detector.get_bounding_box(frame)

                # Filter bboxes with cropper mask
                rbboxes = [bbox for bbox in rbboxes if self.cropper.check_is_valid(*bbox)]

                all_rclasses.append(rclasses)
                all_rscores.append(rscores)
                all_rbboxes.append(rbboxes)
            self.ssd_detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ### CALL INITIAL LABELER ###
            self.init_labeler = InitLabeler(self.config, self.ssd_detector.cap, all_rbboxes, all_rclasses)

            output_count = 0
            frame_skip = 0

            trajectory = []


            ### Second pass: process bounding box data ###
            while self.ssd_detector.cap.isOpened():
                ret, frame = self.ssd_detector.cap.read()
                self.t += 1
                if frame is None:
                    break

                frame_skip += 1
                output_count += 1
                assert num_skip_frames > 0 and int(num_skip_frames) == num_skip_frames
                frame_skip %= num_skip_frames
                if frame_skip != 0:
                    continue

                # Process frame here
                rclasses = all_rclasses[output_count]
                rscores = all_rscores[output_count]
                rbboxes = all_rbboxes[output_count]

                # ###CALL INITIAL LABELER###
                # self.init_labeler(img)

                if self.config.save_images:
                    
                    cv2.imwrite(self.config.save_debug_img_path+'img_'+str(output_count)+'.png',img)
                
                unique, counts = np.unique(rclasses, return_counts=True)
                classes_counts = dict(zip(unique, counts))
                car_count = classes_counts.get(7)
                print "T ", self.t
                current_frame = []

                car_cords = self.get_car_cords(output_count, rclasses, rbboxes)
                ped_cords = self.get_pedestrian_cords(rclasses, rbboxes)

                if output_count > output_limit:
                    break

                for car_cord in car_cords:
                    current_frame.append(car_cord)

                if self.config.use_pedestrian:
                    for ped_cord in ped_cords:
                        current_frame.append(ped_cord)  

                if not len(current_frame) == 0:
                    trajectory.append(current_frame)


            self.ssd_detector.cap.release()
            
            return trajectory


    def get_car_cords(self, frame_i, rclasses, rbboxes):

        frame = []
        for i in range(rclasses.shape[0]):
            if rclasses[i] == 7:
                x_min, y_min, x_max, y_max = rbboxes[i,:]

                if self.cropper.check_is_valid(x_min, y_min, x_max, y_max):

                    ###CHECK IF POINT IS VALID
                    if self.init_labeler.has_init_label(frame_i):
                        if i in self.init_labeler.get_arg_init_label(frame_i):
                            point = ((x_min+x_max)/2.0,y_max, 'car',self.t,'initial_state')
                    else:
                        point = ((x_min+x_max)/2.0,y_max, 'car',self.t,'not_initial_state')

                    frame.append(point)
                else:
                    print "FOUND PARKED CAR"

        return frame


    def get_pedestrian_cords(self,rclasses,rbboxes):
        frame = []
        for i in range(rclasses.shape[0]):
            if rclasses[i] == 15:
                y_min,x_min,y_max,x_max = rbboxes[i,:]

                point = ((x_min+x_max/2.0),(y_min+y_max/2.0), 'pedestrian',self.t)
                frame.append(point)

        return frame



if __name__ == "__main__":
    # main()
    annotateImage('../uds_video_demo/alberta_nobox.png')