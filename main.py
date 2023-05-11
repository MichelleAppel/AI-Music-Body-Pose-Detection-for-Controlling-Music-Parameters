# ------------------------------------------------------------------------------
# AI Music - Body pose detection for controlling music parameters
# Written by Justus Huebotter, Daniel Danilin & Michelle Appel
# ------------------------------------------------------------------------------

import cv2
import argparse
from pose_detector import PoseDetector
from osc_sender import OSCSender

def main():
    # Parse command line arguments - to specify the IP address and port to send OSC messages to
    parser = argparse.ArgumentParser(description="Body pose detection for controlling music parameters.")
    parser.add_argument("--ip", default="127.0.0.1", help="The IP address to send OSC messages to. Default: 127.0.0.1 (localhost)")
    parser.add_argument("--port", type=int, default=7099, help="The port to send OSC messages to. Default: 7099")
    parser.add_argument("--no-osc", action="store_true", help="Disable sending OSC messages. Useful for testing pose detection without sending OSC messages")
    args = parser.parse_args()

    # Initialize pose detector and OSC sender
    pose_detector = PoseDetector()
    if not args.no_osc:
        osc_sender = OSCSender(args.ip, args.port) # IP address and port to send OSC messages to

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Main loop
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect pose and send OSC messages (if enabled)
        pose_data = pose_detector.detect_pose(frame)
        if not args.no_osc:
            for address, value in pose_data.items():
                osc_sender.send_message(address, value) # Send OSC message

        # Display pose in window
        cv2.imshow("Pose Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
