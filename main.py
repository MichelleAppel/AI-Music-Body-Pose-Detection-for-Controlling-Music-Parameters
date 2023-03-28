# ------------------------------------------------------------------------------
# AI Music - Body pose detection for controlling music parameters
# Written by Justus Huebotter, Daniel Danilin & Michelle Appel
# ------------------------------------------------------------------------------

import cv2
import argparse
from pose_detector import PoseDetector
from osc_sender import OSCSender

def main():
    # Argument parsing - because arguing with your code can be frustrating
    parser = argparse.ArgumentParser(description="Body pose detection for controlling music parameters")
    parser.add_argument("--ip", default="127.0.0.1", help="The IP address to send OSC messages to. If you don't know what an IP address is, it's kind of like a street address, but for the internet.")
    parser.add_argument("--port", type=int, default=9000, help="The port to send OSC messages to. A port is like a door on a house - it's where you knock to deliver a message.")
    parser.add_argument("--no-osc", action="store_true", help="Disable sending OSC messages. Use this if you don't want your computer to be a DJ.")
    args = parser.parse_args()

    # Initialize pose detector and OSC sender - to detect poses and send OSC messages respectively
    pose_detector = PoseDetector()
    if not args.no_osc:
        osc_sender = OSCSender(args.ip, args.port)

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

        # Detect pose and send OSC messages - like telekinesis for music parameters.
        pose_data = pose_detector.detect_pose(frame)
        if not args.no_osc:
            for address, value in pose_data.items():
                osc_sender.send_message(address, value)

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
