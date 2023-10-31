# ------------------------------------------------------------------------------
# AI Music - Body pose detection for controlling music parameters
# Written by Justus Huebotter, Daniel Danilin & Michelle Appel
# ------------------------------------------------------------------------------

from ultralytics import YOLO
from models.yolo.body_parts_yolo import BodyParts
from time import time
import numpy as np
import itertools
import copy

class PoseDetector:
    """ Class for detecting body poses in images using YOLOv8n. """
    def __init__(self):
        self.model = YOLO('models/yolo/weights/yolov8n-pose.pt')  # load the YOLO model
        self.model.fuse()

        self.body_parts = BodyParts()

        self.num_humans = 0

        self.prev_pose = None

        self.time = time()
        self.prev_time = None

    def detect_pose(self, image):
        """ Detect body poses in an image and return pose data. 
        
        Args:
            image (np.ndarray): Image to detect poses in.
            
        Returns:
            pose_data (dict): Dictionary containing pose data.
            annotated_frame (np.ndarray): Image with annotated pose landmarks.
        
        """
        self.time = time()

        # Process image and detect pose landmarks
        results = copy.deepcopy(self.model(image)[0].cpu().numpy())
        sorted_indices = self.sort_results(results, order="left_to_right")  # or "up_to_down"
    
        # Set names for humans
        for i in range(len(results.boxes.cls)):
            results.boxes.cls[i] = sorted_indices[i]
            results.names[sorted_indices[i]] = f"person {sorted_indices[i]}"

        # Initialize empty dictionary for pose data
        pose_data = {}
        
        self.num_humans = len(results)
        pose_data["/num_humans"] = self.num_humans

        if self.num_humans > 0:
            # Calculate pose data
            for human_idx in sorted_indices:
                result = results[human_idx]

                # Calculate pose data
                human_pose = self.calculate_pose(result, human_idx)
                pose_data.update(human_pose)

                # Calculate normalized pose data
                normalized_pose = self.calculate_normalized_pose(result, human_idx)
                pose_data.update(normalized_pose)

                # Calculate pose velocity
                pose_velocity = self.calculate_pose_velocity(normalized_pose, human_idx)
                pose_data.update(pose_velocity)

            if self.num_humans > 0:
                # Calculate pose distance between humans
                pose_distance = self.calculate_pose_distance(pose_data)
                pose_data.update(pose_distance)

        # Draw pose landmarks on image
        annotated_frame = results.plot()

        # Set previous pose as current pose
        self.prev_pose = pose_data
        self.prev_time = self.time

        # Return pose data dictionary
        return pose_data, annotated_frame

    def sort_results(self, results, order="left_to_right", rule="left_corners"):
        """ Sort the YOLO results based on the given order. 
        
        Args:
            results (YOLOv8Result): The results object from YOLOv8 model.
            order (str): The order for sorting. Can be "left_to_right" or "up_to_down".
            rule (str): The rule for sorting. Can be "left_corners", "right_corners" or "centers".
            
        Returns:
            sorted_results (YOLOv8Result): The sorted results object.
        """
        
        # Extract bounding box data
        bboxes = results.boxes.xyxyn

        # Calculate the corners of the bounding boxes
        corners = [(bbox[0], bbox[1]) for bbox in bboxes]
        
        # Calculate center of the bounding boxes
        centers = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
        
        # Calculate area of the bounding boxes
        areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
        
        # Sort based on the desired order
        if order == "left_to_right":
            if rule == "left_corners":
                sorted_indices = np.argsort([c[0] for c in corners])
            elif rule == "right_corners":
                sorted_indices = np.argsort([c[1] for c in corners])
            elif rule == "centers":
                sorted_indices = np.argsort(centers)
            else:
                raise NotImplementedError(f"Rule {rule} not implemented!")
        elif order == "up_to_down":
            sorted_indices = np.argsort([bbox[1] for bbox in bboxes])
        elif order == "big_to_small":
            sorted_indices = np.argsort(areas)[::-1]  # Inverse to sort from big to small

        else:
            raise ValueError("Invalid sort order!")
        
        return sorted_indices

    def calculate_pose(self, result, human_index=0):
        """ Calculate pose data from a YOLOv8n result.

        Args:
            result (YOLOv8Result): Result from YOLOv8 model.
            human_index (int): Index of human to calculate pose data for.

        Returns:
            pose (dict): Dictionary containing pose data.
            
        """
        
        # Get keypoints from result
        keypoints = result.keypoints.xyn[0]

        # Initialize empty dictionary for pose data
        pose = {}

        # Iterate through detected body parts and add pose data to pose dictionary
        for index in self.body_parts.body_parts.keys():
            keypoint = keypoints[index]
            # Add pose data to pose dictionary
            x, y = keypoint
            x, y = x.item(), y.item()

            if x == 0 and y == 0:
                pass
            else:
                x = 1 - x
                y = 1 - y

            pose[f"/{human_index}/{self.body_parts.body_parts[index]}/pose/x"] = x
            pose[f"/{human_index}/{self.body_parts.body_parts[index]}/pose/y"] = y

        # Return pose data dictionary
        return pose

    def calculate_normalized_pose(self, result, human_index=0):
        """ Calculate normalized pose data from a YOLOv8n result. 
        
        Args:
            result (YOLOv8Result): Result from YOLOv8n model.
            human_index (int): Index of human to calculate pose data for.
            
        Returns:
            pose (dict): Dictionary containing normalized pose data.
            
        """

        keypoints = result.keypoints.xyn[0]
        bbox = result.boxes.xyxyn[0]

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
            normalized_keypoint = (relative_keypoint / np.array([bbox_width, bbox_height])) + 0.5

            # Add pose data to pose dictionary
            x, y = normalized_keypoint
            pose[f"/{human_index}/{self.body_parts.body_parts[index]}/normalized_pose/x"] = x.item()
            pose[f"/{human_index}/{self.body_parts.body_parts[index]}/normalized_pose/y"] = y.item()

        # Return pose data dictionary
        return pose

    def calculate_pose_velocity(self, normalized_pose, human_index=0):
        """ Calculate pose velocity from a normalized pose.

        Args:
            normalized_pose (dict): Dictionary containing normalized pose data.
            human_index (int): Index of human to calculate pose data for.

        Returns:
            pose_velocity (dict): Dictionary containing pose velocity data.

        """

        # Get the last pose
        prev_pose = self.prev_pose

        if prev_pose is None:
            return {}

        # Convert the prev_pose dictionary to a numpy array
        prev_keypoints_np = {}
        for index, part_name in self.body_parts.body_parts.items():
            key_x = f"/{human_index}/{part_name}/normalized_pose/x"
            key_y = f"/{human_index}/{part_name}/normalized_pose/y"
            
            if key_x in prev_pose and key_y in prev_pose:
                prev_keypoints_np[index] = np.array((prev_pose[key_x], prev_pose[key_y]))
            else:
                # Handle missing key, for example, by skipping this body part
                continue
        # prev_keypoints_np = np.array(prev_keypoints_np)
        
        if len(prev_keypoints_np) == 0:
            return {}
        
        # Convert the normalized_pose dictionary to a numpy array
        keypoints_np = {}
        for index, part_name in self.body_parts.body_parts.items():
            key_x = f"/{human_index}/{part_name}/normalized_pose/x"
            key_y = f"/{human_index}/{part_name}/normalized_pose/y"
            
            if key_x in normalized_pose and key_y in normalized_pose:
                keypoints_np[index] = np.array((normalized_pose[key_x], normalized_pose[key_y]))
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

    def calculate_pose_distance(self, pose_data):
        """ Calculate the distance between all the body parts of all humans in the frame. 
        
        Args:
            pose_data (dict): Dictionary containing pose data.
            
        Returns:
            pose_distance (dict): Dictionary containing pose distance data.
            
        """

        # Initialize empty dictionary for pose distance data
        pose_distance = {}

        # Iterate through detected body parts and add pose distance data to pose distance dictionary
        for human_1, human_2 in itertools.product(range(self.num_humans), repeat=2):
            for _, part_name1 in self.body_parts.body_parts.items():
                for _, part_name2 in self.body_parts.body_parts.items():
                    key1_x = f"/{human_1}/{part_name1}/pose/x"
                    key1_y = f"/{human_1}/{part_name1}/pose/y"
                    key2_x = f"/{human_2}/{part_name2}/pose/x"
                    key2_y = f"/{human_2}/{part_name2}/pose/y"
                    if key1_x in pose_data and key1_y in pose_data and key2_x in pose_data and key2_y in pose_data:
                        x1, y1 = pose_data[key1_x], pose_data[key1_y]
                        x2, y2 = pose_data[key2_x], pose_data[key2_y]
                        pose_distance[f"/{human_1}/{part_name1}/distance/{human_2}/{part_name2}"] = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

        # Return pose distance data dictionary
        return pose_distance
