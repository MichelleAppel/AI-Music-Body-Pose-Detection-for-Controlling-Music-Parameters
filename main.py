# ------------------------------------------------------------------------------
# AI Music - Body pose detection for controlling music parameters
# Written by Justus Huebotter, Daniel Danilin & Michelle Appel
# ------------------------------------------------------------------------------

import cv2
import argparse
from models.yolo.pose_detector_yolo import PoseDetector
from osc_sender import OSCSender

import time

max_FPS = 30

FRAME_WIDTH = 960
FRAME_HEIGHT = 1706

def main():
    # Parse command line arguments - to specify the IP address and port to send OSC messages to
    parser = argparse.ArgumentParser(description="Body pose detection for controlling music parameters.")
    parser.add_argument("--ip", default="127.0.0.1", help="The IP address to send OSC messages to. Default: 127.0.0.1 (localhost)")
    parser.add_argument("--port", type=int, default=7099, help="The port to send OSC messages to. Default: 7099")
    parser.add_argument("--no-osc", action="store_true", help="Disable sending OSC messages. Useful for testing pose detection without sending OSC messages")
    parser.add_argument("--image", type=str, help="Path to the source image for debugging") # New command line argument
    parser.add_argument("--video", type=str, help="Path to the source video")
    parser.add_argument("--cam", type=int, default=0, help="Camera input number")

    args = parser.parse_args()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, max_FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    # Initialize pose detector and OSC sender
    pose_detector = PoseDetector()
    if not args.no_osc:
        osc_sender = OSCSender(args.ip, args.port) # IP address and port to send OSC messages to

    # Open webcam or video file
    if args.image:
        frame = cv2.imread(args.image)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        # Max fps
        cap.set(cv2.CAP_PROP_FPS, max_FPS)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
        cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)
    else:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)

    t0 = time.time()
    # Main loop
    while True:
        # Capture frame from webcam
        if args.image:
            if not frame.any():
                print("Error: Could not load image.")
                break
        elif args.video or not args.image:
            # Calculate frame according to time
            # t = time.time()
            # frame = int((t-t0)*max_FPS)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = cap.read()

            if not ret:
                break
                # t0 = time.time()

            # Flip frame horizontally for mirror effectq
            # frame = cv2.flip(frame, 1)

        # Detect pose and send OSC messages (if enabled)
        pose_data, annotated_frame = pose_detector.detect_pose(frame)
        if args.image:
            print(pose_data)

        if not args.no_osc:
            for address, value in pose_data.items():
                osc_sender.send_message(address, value) # Send OSC message

        # Display pose in window
        out.write(annotated_frame)
        cv2.imshow("Pose Detection", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # t1 = time.time()

        # tt = 1/max_FPS
        # if t1-t0 < tt:
        #     time.sleep(tt-(t1-t0))
        
        # print(f"FPS: {1/(t1-t0)}")

    # Release webcam and close window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
