# ------------------------------------------------------------------------------
# AI Music - Body pose detection for controlling music parameters
# Written by Justus Huebotter, Daniel Danilin & Michelle Appel
# ------------------------------------------------------------------------------

import cv2
import mediapipe as mp
from body_parts import BodyParts
import numpy as np


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.body_parts = BodyParts()
        self.prev_pose = None

    def detect_pose(self, image):
        # Process image and detect pose landmarks
        results = self.pose.process(image)

        # Initialize empty dictionary for pose data
        pose_data = {}

        # Calculate pose data
        pose = self.calculate_pose(results.pose_landmarks)
        pose_data.update(pose)

        # Calculate velocity data
        if self.prev_pose is not None:
            velocity = self.calculate_velocity(results.pose_landmarks)
            pose_data.update(velocity)

        # Draw pose landmarks on image
        self.draw_pose(image, results.pose_landmarks)

        # Set previous pose as current pose
        self.prev_pose = results.pose_landmarks

        # Return pose data dictionary
        return pose_data

    def calculate_pose(self, landmarks):
        # Initialize empty dictionary for pose data
        pose = {}

        if landmarks:
            # Iterate through detected body parts and add pose data to pose dictionary
            for index in self.body_parts.body_parts.keys():
                landmark = landmarks.landmark[index]
                if landmark.visibility > 0:
                    # Add pose data to pose dictionary
                    x, y, z = landmark.x, landmark.y, landmark.z
                    pose[f"/{self.body_parts.body_parts[index]}/pose/x"] = x
                    pose[f"/{self.body_parts.body_parts[index]}/pose/y"] = y
                    pose[f"/{self.body_parts.body_parts[index]}/pose/z"] = z

        # Return pose data dictionary
        return pose
    

    def calculate_velocity(self, current_landmarks):
        # Initialize empty dictionary for velocity data
        velocity = {}

        if current_landmarks is None:
            # No landmarks detected in current frame, return empty velocity dictionary
            return velocity

        # Iterate through detected body parts and calculate velocity between current and previous landmark position
        for index in self.body_parts.body_parts.keys():
            current_landmark = current_landmarks.landmark[index]
            prev_landmark = self.prev_pose.landmark[index]
            if current_landmark.visibility > 0 and prev_landmark.visibility > 0:
                # Calculate velocity and add to velocity dictionary
                address = f"/{self.body_parts.body_parts[index]}/velocity"
                velocity[address] = self.calculate_velocity_vector(current_landmark, prev_landmark)

        # Return velocity data dictionary
        return velocity


    def calculate_velocity_vector(self, current_landmark, prev_landmark):
        # Calculate velocity between current and previous landmark position
        current_point = np.array([current_landmark.x, current_landmark.y, current_landmark.z])
        prev_point = np.array([prev_landmark.x, prev_landmark.y, prev_landmark.z])
        distance = np.linalg.norm(current_point - prev_point)

        # Time is relative
        time_elapsed = 1.0
        velocity = distance / time_elapsed

        # If you're moving too fast or too slow, we'll still report your velocity
        # But if your landmark visibility is negative, we'll assume you're a ghost and ignore you
        if current_landmark.visibility < 0 or prev_landmark.visibility < 0:
            return 0

        # Return velocity vector
        return velocity

    
    def draw_pose(self, image, landmarks, display_text=True):
        """
        Draw pose landmarks on the input image and display corresponding body part names and coordinates as text.

        :param image: The input image to draw landmarks on.
        :param landmarks: The detected pose landmarks.
        :param display_text: Boolean flag to determine whether to display text with body part names and coordinates.
        :return: The input image with landmarks and text overlay.
        """

        # When no landmarks are detected
        if landmarks is None:
            # Create an empty image with the same size as the input image to draw the semi-transparent overlay on
            overlay_image = np.zeros_like(image)
            # Add a semi-transparent overlay to the input image for visualization
            alpha = 0.5
            cv2.addWeighted(overlay_image, alpha, image, 1 - alpha, 0, image)
            return image

        # Initialize drawing utility
        mp_drawing = mp.solutions.drawing_utils

        # Create an empty image with the same size as the input image to draw landmarks on
        annotated_image = np.zeros_like(image)

        # Loop through all the detected body parts and draw corresponding landmarks and text
        for index in self.body_parts.body_parts.keys():
            landmark = landmarks.landmark[index]
            if landmark.visibility > 0:
                x, y, z = landmark.x, landmark.y, landmark.z

                # Create a list of text lines to display for each body part
                text_lines = [
                    f"{self.body_parts.body_parts[index]}",
                    f"x: {x:.2f}",
                    f"y: {y:.2f}",
                    f"z: {z:.2f}",
                ]

                # Calculate the y-coordinate offset for each text line
                y_offset = int(y * image.shape[0])

                # Loop through all the text lines and display them on the annotated image
                for i, line in enumerate(text_lines):
                    cv2.putText(annotated_image, line, (int(x * image.shape[1]), y_offset + i*15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

                # Draw landmarks on the annotated image
                mp_drawing.draw_landmarks(
                    annotated_image,
                    landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

        # Add a semi-transparent overlay of the annotated image on the input image for visualization
        alpha = 0.5
        cv2.addWeighted(annotated_image, alpha, image, 1 - alpha, 0, image)

        # Return the input image with landmarks and text overlay
        return image
