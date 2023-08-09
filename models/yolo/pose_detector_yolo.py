# ------------------------------------------------------------------------------
# AI Music - Body pose detection for controlling music parameters
# Written by Justus Huebotter, Daniel Danilin & Michelle Appel
# ------------------------------------------------------------------------------

from ultralytics import YOLO
from models.yolo.body_parts_yolo import BodyParts
from time import time
import numpy as np

class PoseDetector:
    def __init__(self):
        self.model = YOLO('models/yolo/weights/yolov8n-pose.pt')  # load the YOLO model
        self.body_parts = BodyParts()

        self.prev_pose = None

        self.time = time()
        self.prev_time = None

    def detect_pose(self, image):
        self.time = time()

        # Process image and detect pose landmarks
        results = self.model(image)

        # Initialize empty dictionary for pose data
        pose_data = {}
        
        # Calculate pose data
        for human in range(len(results[0].keypoints.xyn)):
            result = results[0].keypoints.xyn[human].cpu().numpy()

            if len(results[0].boxes.xyxyn) > 0:
                bbox = results[0].boxes.xyxyn[human].cpu().numpy()
            else:
                bbox = None

            if len(result) > 0:
                human_pose = self.calculate_pose(result, human)
                relative_pose = self.calculate_normalized_pose(result, bbox, human)
                pose_velocity = self.calculate_pose_velocity(relative_pose, human)

                pose_data.update(human_pose)
                pose_data.update(relative_pose)
                pose_data.update(pose_velocity)

        # Draw pose landmarks on image
        annotated_frame = results[0].plot()

        # Set previous pose as current pose
        self.prev_pose = pose_data
        self.prev_time = self.time

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

    def calculate_normalized_pose(self, keypoints, bbox, human_index=0):
        if bbox is None:
            return {}

        # Initialize empty dictionary for pose data
        pose = {}

        # Calculate bounding box center
        bbox_center_x = (bbox[0] + bbox[2]) / 2 
        bbox_center_y = (bbox[1] + bbox[3]) / 2 

        # Calculate bounding box dimensions for normalization (if necessary)
        bbox_width = (bbox[2] - bbox[0]) 
        bbox_height = (bbox[3] - bbox[1])

        # Iterate through detected body parts and add pose data to pose dictionary
        for index in self.body_parts.body_parts.keys():
            keypoint = keypoints[index]

            # Calculate relative pose with respect to bounding box center
            relative_keypoint = keypoint - np.array([bbox_center_x, bbox_center_y])

            # Normalize relative pose by bounding box dimensions (optional)
            normalized_keypoint = relative_keypoint / np.array([bbox_width, bbox_height])

            # Add pose data to pose dictionary
            x, y = normalized_keypoint
            pose[f"/{human_index}/{self.body_parts.body_parts[index]}/normalized_pose/x"] = x.item()
            pose[f"/{human_index}/{self.body_parts.body_parts[index]}/normalized_pose/y"] = y.item()

        # Return pose data dictionary
        return pose

    
    def calculate_pose_velocity(self, normalized_pose, human_index=0):
        # Get the last pose
        prev_pose = self.prev_pose

        if prev_pose is None:
            return {}

        # Convert the prev_pose dictionary to a numpy array
        prev_keypoints_np = []
        for index, part_name in self.body_parts.body_parts.items():
            key_x = f"/{human_index}/{part_name}/normalized_pose/x"
            key_y = f"/{human_index}/{part_name}/normalized_pose/y"
            
            if key_x in prev_pose and key_y in prev_pose:
                prev_keypoints_np.append([prev_pose[key_x], prev_pose[key_y]])
            else:
                # Handle missing key, for example, by skipping this body part
                continue
        prev_keypoints_np = np.array(prev_keypoints_np)
        
        if len(prev_keypoints_np) == 0:
            return {}
        
        # Convert the normalized_pose dictionary to a numpy array
        keypoints_np = []
        for index, part_name in self.body_parts.body_parts.items():
            key_x = f"/{human_index}/{part_name}/normalized_pose/x"
            key_y = f"/{human_index}/{part_name}/normalized_pose/y"
            
            if key_x in normalized_pose and key_y in normalized_pose:
                keypoints_np.append([normalized_pose[key_x], normalized_pose[key_y]])
            else:
                # Handle missing key, for example, by skipping this body part
                continue

        if len(keypoints_np) == 0:
            return {}
        
        # Initialize empty dictionary for pose data
        velocity = {}

        # Iterate through detected body parts and calculate velocity
        for index, part_name in self.body_parts.body_parts.items():
            keypoint_x, keypoint_y = keypoints_np[index]
            prev_keypoint_x, prev_keypoint_y = prev_keypoints_np[index]

            # Calculate velocity given the prev time and current time
            vx = (keypoint_x - prev_keypoint_x) / (self.time - self.prev_time)
            vy = (keypoint_y - prev_keypoint_y) / (self.time - self.prev_time)
            
            # Add pose data to velocity dictionary
            velocity[f"/{human_index}/{part_name}/velocity/x"] = abs(vx)
            velocity[f"/{human_index}/{part_name}/velocity/y"] = abs(vy)
            velocity[f"/{human_index}/{part_name}/velocity/xy"] = (vx**2 + vy**2)**0.5

        # Return pose data dictionary
        return velocity
