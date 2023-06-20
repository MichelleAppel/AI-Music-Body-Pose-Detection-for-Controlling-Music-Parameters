# ------------------------------------------------------------------------------
# AI Music - Body pose detection for controlling music parameters
# Written by Justus Huebotter, Daniel Danilin & Michelle Appel
# ------------------------------------------------------------------------------

from ultralytics import YOLO
import cv2
import numpy as np
from body_parts_yolo import BodyParts
import supervision as sv

class PoseDetector:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')  # load the YOLO model
        self.body_parts = BodyParts()
        self.prev_pose = None

    def detect_pose(self, image):
        # Process image and detect pose landmarks
        results = self.model(image)

        # Initialize empty dictionary for pose data
        pose_data = {}
        
        # Calculate pose data
        for human in range(len(results[0].keypoints.xyn)):
            result = results[0].keypoints.xyn[human]
            if len(result) > 0:
                human_pose = self.calculate_pose(result, human)
                pose_data.update(human_pose)

        # Draw pose landmarks on image
        annotated_frame = results[0].plot()

        # Set previous pose as current pose
        # self.prev_pose = pose_data

        # Return pose data dictionary
        return pose_data, annotated_frame

    def calculate_pose(self, keypoints, human_index=0):
        # Initialize empty dictionary for pose data
        pose = {}

        # Iterate through detected body parts and add pose data to pose dictionary
        for index in self.body_parts.body_parts.keys():
            keypoint = keypoints[index]
            # Add pose data to pose dictionary
            x, y = keypoint
            pose[f"/{human_index}/{self.body_parts.body_parts[index]}/pose/x"] = x.item()
            pose[f"/{human_index}/{self.body_parts.body_parts[index]}/pose/y"] = y.item()

        # Return pose data dictionary
        return pose
