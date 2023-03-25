# AI Music - Body Pose Detection for Controlling Music Parameters

## Introduction

This project uses body pose detection to control music parameters, such as tempo or volume, based on the user's movements. It uses the Mediapipe library to detect the user's body landmarks and calculate their velocity over time. The resulting pose data is then used to send OSC messages to a music software, such as Ableton Live or Max/MSP.

## Installation

To install the required libraries, run the following command:

`pip install -r requirements.txt`

## Usage

To use the program, run the following command:

`python main.py [--ip IP_ADDRESS] [--port PORT_NUMBER] [--no-osc]`

Optional arguments:

-   `--ip IP_ADDRESS`: the IP address to send OSC messages to (default: 127.0.0.1)
-   `--port PORT_NUMBER`: the port number to send OSC messages to (default: 9000)
-   `--no-osc`: disable sending OSC messages (default: False)

The program will open a video stream from the default camera and show the user's pose landmarks in real time. It will also send OSC messages to the specified IP address and port number if enabled. To quit the program, press the `q` key.

## Customization

If you want to customize the program, you can modify the following files:

-   `pose_detector.py`: This file contains the `PoseDetector` class, which is responsible for detecting body pose landmarks and calculating pose data (including velocity). You can modify this class to change how pose data is detected and calculated.
-   `osc_sender.py`: This file contains the `OSCSender` class, which is responsible for sending OSC messages to a specified IP address and port. You can modify this class to change how OSC messages are sent (for example, to control different music parameters).
-   `body_parts.py`: This file contains a `BodyParts` class, which defines the body parts that are detected by the `PoseDetector` class. You can modify this class to add or remove body parts as needed.
-   `util.py`: This file contains utility functions used by the other files. You can modify this file to add your own utility functions.

## Contributing

If you find a bug or want to suggest a new feature, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
