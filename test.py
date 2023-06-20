from ultralytics import YOLO
import cv2

def main():
    # Load a model
    print("Loading model...")
    model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)
    print("Model loaded!")

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Main loop
    while cap.isOpened():
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1) 
        
        # Use the model
        results = model(frame)
        annotated_frame = results[0].plot()
        keypoints = results[0].keypoints.xyn[0]


        # Display pose in window
        cv2.imshow("Pose Detection", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close window
    cap.release()
    cv2.destroyAllWindows()


main()